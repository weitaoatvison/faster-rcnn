
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <ctime>
#include <cassert>
#include <algorithm>

#include <ctype.h>
#include <sys/time.h>

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;
using namespace std;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<int, float> Prediction;

class CaffeClassifier {
 public:
  CaffeClassifier(const string& model_file,
             const string& trained_file,
             const bool use_GPU,
             const int batch_size);

  vector< vector<Prediction> > ClassifyBatch(const vector<Mat> imgs, int num_classes);
  vector<Blob<float>* > PredictBatch(vector<Mat> imgs, float a, float b, float c);

  void SetMean(float a, float b, float c);
 private:

  void WrapBatchInputLayer(vector<vector<Mat> > *input_batch);

  void PreprocessBatch(const vector<Mat> imgs, vector<vector<Mat> >* input_batch);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  int batch_size_;
  cv::Mat mean_;
  bool useGPU_;
};

CaffeClassifier::CaffeClassifier(const string& model_file,
                       const string& trained_file,
                       const bool use_GPU,
                       const int batch_size) {
   if (use_GPU) {
       Caffe::set_mode(Caffe::GPU);
       Caffe::SetDevice(0);
       useGPU_ = true;
   }
   else {
       Caffe::set_mode(Caffe::CPU);
       useGPU_ = false;
   }

  /* Set batchsize */
  batch_size_ = batch_size;

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  //CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

//std::vector< float >  CaffeClassifier::PredictBatch(const vector< cv::Mat > imgs) {
vector<Blob<float>* > CaffeClassifier::PredictBatch(vector< cv::Mat > imgs, float a, float b, float c) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  
  input_geometry_.height = imgs[0].rows;
  input_geometry_.width = imgs[0].cols;
  input_layer->Reshape(batch_size_, num_channels_,
                       input_geometry_.height,
                       input_geometry_.width);
  
  float* input_data = input_layer->mutable_cpu_data();
  int cnt = 0;
  for(int i = 0; i < imgs.size(); i++) {
    cv::Mat sample;
    cv::Mat img = imgs[i];

    if (img.channels() == 3 && num_channels_ == 1)
      cv::cvtColor(img, sample, CV_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
      cv::cvtColor(img, sample, CV_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
      cv::cvtColor(img, sample, CV_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
      cv::cvtColor(img, sample, CV_GRAY2BGR);
    else
      sample = img;

    if((sample.rows != input_geometry_.height) || (sample.cols != input_geometry_.width)) {
        cv::resize(sample, sample, Size(input_geometry_.width, input_geometry_.height));
    }

    for(int k = 0; k < sample.channels(); k++) {
        for(int i = 0; i < sample.rows; i++) {
            for(int j = 0; j < sample.cols; j++) {
               input_data[cnt] = (float(sample.at<uchar>(i,j*3+k))-128);
               cnt += 1;
            }
        }
    }
  }
  /* Forward dimension change to all layers. */
  net_->Reshape();
 
  struct timeval start;
  gettimeofday(&start, NULL);

  net_->ForwardPrefilled();

  if(useGPU_) {
    cudaDeviceSynchronize();
  }

  struct timeval end;
  gettimeofday(&end, NULL);
  cout << "pure model predict time cost: " << (1000000*(end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec)/1000 << endl;

  /* Copy the output layer to a std::vector */
  vector<Blob<float>* > outputs;
  for(int i = 0; i < net_->num_outputs(); i++) {
    Blob<float>* output_layer = net_->output_blobs()[i];
    outputs.push_back(output_layer);
  }
  return outputs;
}

struct Bbox {
    float confidence;
    Rect rect;
    Rect gt;
    bool deleted;

};
bool mycmp(struct Bbox b1, struct Bbox b2) {
    return b1.confidence > b2.confidence;
}

void nms(vector<struct Bbox>& p, float threshold) {
   sort(p.begin(), p.end(), mycmp);
   int cnt = 0;
   for(int i = 0; i < p.size(); i++) {
     if(p[i].deleted) continue;
     cnt += 1;
     for(int j = i+1; j < p.size(); j++) {
       if(!p[j].deleted) {
         cv::Rect intersect = p[i].rect & p[j].rect;
         float iou = intersect.area() * 1.0 / (p[i].rect.area() + p[j].rect.area() - intersect.area());
         if (iou > threshold) {
           p[j].deleted = true;
         }
       }
     }
   }
   //cout << "[after nms] left is " << cnt << endl;
}

vector<struct Bbox> get_final_bbox(vector<Mat> images, vector<Blob<float>* >& outputs, int class_index, vector<vector<float> > areas, vector<vector<float> > ratios, float global_confidence, float enlarge_ratio, int target_min_size, float stride) {

    Blob<float>* cls = outputs[2*class_index];
    Blob<float>* reg = outputs[2*class_index + 1];
    
    assert(areas.size() == ratios.size());

    int scale_num = 0;
    for(int s = 0; s < areas.size(); s++) {
        assert(areas[s].size() == ratios[s].size());
        scale_num += areas[s].size() * ratios[s].size();
    }

    assert((cls_num() * cls_channels()) / scale_num == 2);
    assert((reg_num() * reg_channels()) / scale_num == 4);
    cout << "[debug num, channels, height, width]" << cls->num() << " " << cls->channels() << " " << cls->height() << " " << cls->width() << endl;
    cout << "[scale_num]" << scale_num << endl;

    cls->Reshape(cls->num()*scale_num, cls->channels()/scale_num, cls->height(), cls->width());
    reg->Reshape(reg->num()*scale_num, reg->channels()/scale_num, reg->height(), reg->width());

    cout << "[debug id, stride, h,w] " << class_index << " " <<  stride << " " << cls->height()  << " " << cls->width()<< endl;
    fflush(stdout);

    assert(cls->num() == reg->num() && cls->num() == scale_num);

    assert(cls->channels() == 2);
    assert(reg->channels() == 4);

    assert(cls->height() == reg->height());
    assert(cls->width() == reg->width());
    
    vector<struct Bbox> vbbox;
    //int cls_cnt = 0;
    //int reg_cnt = 0;
    const float* cls_cpu = cls->cpu_data();
    const float* reg_cpu = reg->cpu_data();

    //float mean[4] = {0, 0, 0, 0};
    //float std[4] = {1,1,1,1};

    float *gt_ww = new float[scale_num];
    float *gt_hh = new float[scale_num];
    //float global_ratio = 1.0 * min(images[0].rows, images[0].cols) / target_min_size;

    int cnt = 0;
    for(int s = 0; s < areas.size(); s++) {
        for(int i = 0; i < areas[s].size(); i++) {
            for(int j = 0; j < ratios[s].size(); j++) {
               gt_ww[cnt] = sqrt(areas[s][i] * ratios[s][j]); //* global_ratio;
               gt_hh[cnt] = gt_ww[cnt] / ratios[s][j]; // * global_ratio;
               cnt++; 
            }
        }
    }

    int cls_index = 0;
    int reg_index = 0;
    for(int i = 0; i < cls->num(); i++) {  // = batchsize * 25
        //float confidence = 0;
         { // = 2
            int skip = cls->height() * cls->width();
            for(int h = 0; h < cls->height(); h++) {
                for(int w = 0; w < cls->width(); w++) {
                    float confidence;
                    float rect[4] = {};
                    float gt_cx = w * stride; 
                    float gt_cy = h * stride;
                    {
                        float x0 = cls_cpu[cls_index];
                        float x1 = cls_cpu[cls_index + skip];
                        float min_01 = min(x1, x0);
                        x0 -= min_01;
                        x1 -= min_01;
                        confidence = exp(x1)/(exp(x1)+exp(x0));
                    } 
                    if(confidence > global_confidence){

                        for(int j = 0; j < 4; j++) {
                            rect[j] = reg_cpu[reg_index + j*skip];
                        }
                        
                        float shift_x = 0, shift_y = 0;
                        //shift_x = w * 16 + 16/2 -1 - gt_ww[i]/2;
                        //shift_y = h * 16 + 16/2 -1 - gt_hh[i]/2;
                        shift_x = w * stride + stride / 2 - 1;
                        shift_y = h * stride + stride / 2 - 1;
                        rect[2] = exp(rect[2]) * gt_ww[i];
                        rect[3] = exp(rect[3]) * gt_hh[i];
                        //rect[0] = rect[0] * gt_ww[i] + gt_ww[i]/2 - 0.5*rect[2] + shift_x;
                        //rect[1] = rect[1] * gt_hh[i] + gt_hh[i]/2 - 0.5*rect[3] + shift_y;
                        rect[0] = rect[0] * gt_ww[i] - 0.5*rect[2] + shift_x;
                        rect[1] = rect[1] * gt_hh[i] - 0.5*rect[3] + shift_y;

                        struct Bbox bbox;
                        bbox.confidence = confidence;
                        bbox.rect = Rect(rect[0], rect[1], rect[2], rect[3]);
                        assert(images.size() == 1);
                        bbox.rect &= Rect(0,0,images[0].cols, images[0].rows);
                        bbox.deleted = false;
                        bbox.gt = Rect(gt_cx - gt_ww[i]/2, gt_cy-gt_hh[i]/2, gt_ww[i], gt_hh[i]) & Rect(0,0,images[0].cols, images[0].rows);
                        vbbox.push_back(bbox);
                    }

                    cls_index += 1;
                    reg_index += 1;
                }
            }
            cls_index += skip;
            reg_index += 3*skip;
        }
    }
    
    sort(vbbox.begin(), vbbox.end(), mycmp);

    nms(vbbox, 0.5);
    
    cout << "[debug nms passed!]\n" << endl;
    vector<struct Bbox> final_vbbox;
    for(int i = 0; i < vbbox.size(); i++) {
        if(!vbbox[i].deleted && vbbox[i].confidence > global_confidence) {
            //Rect box = vbbox[i].rect;
            struct Bbox box = vbbox[i];
            float x = box.rect.x / enlarge_ratio;
            float y = box.rect.y / enlarge_ratio;
            float w = box.rect.width / enlarge_ratio;
            float h = box.rect.height / enlarge_ratio;
            box.rect.x = x;
            box.rect.y = y;
            box.rect.width = w;
            box.rect.height = h;
            if (vbbox[i].confidence > global_confidence){
                final_vbbox.push_back(box);
            }
        }
    }
    delete [] gt_ww;
    delete [] gt_hh;
    return final_vbbox;
}

int main(int argc, char** argv) {

  google::InitGoogleLogging(argv[0]);

  // caffe variables
  string prefix = ""; 
  string model_file   = argv[1]; //prefix + "/mnt/ssd/DeepV/faster_rcnn/detector/test.prototxt"; //  rcnn_googlenet.prototxt
  string trained_file = argv[2]; //prefix + "/mnt/ssd/DeepV/faster_rcnn/detector/googlenet_faster_rcnn_iter_48000.caffemodel"; // rcnn_googlenet.caffemodel
  
  //////////////////
  // string image_list = "/mnt/ssd/DeepV/roi/nightfront_v1.3/truckData_aug.txt";
  string image_list = argv[3]; // "/mnt/ssd/DeepV/roi/nightfront_v1.3/truckData_aug.txt"; //
  //////////////////

  CaffeClassifier classifier_single(model_file, trained_file, true, 1);

  model_file = argv[4];//"/mnt/ssd/DeepV/faster_rcnn/detector/deploy.prototxt"; // car_post_cls.prototxt
  trained_file = argv[5]; //"/mnt/ssd/DeepV/faster_rcnn/detector/car_not_car_train_iter_30000.caffemodel"; // car_post_cls.caffemodel

  //CaffeClassifier cls_single(model_file, trained_file, true, 1);
  FILE *fcin  = fopen(image_list.c_str(),"r");
  if(!fcin) {
    cout << "can not open filelist" << endl;
  }
  char image_filename[200];

  cout << "single mode" << endl;

  int tot_cnt = 0;

  //////////////////
  // string output_list = "/mnt/ssd/DeepV/roi/nightfront_v1.3/truckData_aug.output";
  string output_list = argv[6]; // "/mnt/ssd/DeepV/roi/nightfront_v1.3/truckData_aug.output";
  FILE* fid = fopen(output_list.c_str(), "w");
  //////////////////

  float global_confidence = atof(argv[7]);
  while(fscanf(fcin, "%s", image_filename)!=EOF) {
        
        struct timeval start;
        gettimeofday(&start, NULL);
        tot_cnt += 1;
        //if(tot_cnt < 28380) continue;
	fprintf(fid, "%s ", image_filename);
        cout << "filename " << string(image_filename) << endl;
        vector<Mat> images;
        Mat image = imread(image_filename, -1);
        if (image.empty()) {
            cout << "Wrong Image" << endl;
            continue;
        }
        
         
        cv::cvtColor(image, image, CV_BGR2RGB);
        //resize(image, image, Size(600, int(image.rows*1.0/image.cols*600)));
        //struct timeval end_0;
        //gettimeofday(&end_0, NULL);
        //cout << "readling image time cost: " << (1000000*(end_0.tv_sec - start.tv_sec) + end_0.tv_usec - start.tv_usec)/1000 << endl;

        Mat img;

        float border_ratio = 0.00;

        img = image.clone();

        int max_size = max(img.rows, img.cols);
        int min_size = min(img.rows, img.cols);
        float target_min_size = atof(argv[8]);
        float target_max_size = 1000.0;
        float enlarge_ratio = target_min_size / min_size;

        if(max_size * enlarge_ratio > target_max_size) {
            enlarge_ratio = target_max_size / max_size;
        }

        int target_row = img.rows * enlarge_ratio;
        int target_col = img.cols * enlarge_ratio;

        resize(img, img, Size(target_col, target_row));

        images.push_back(img);

        //struct timeval end_1;
        //gettimeofday(&end_1, NULL);
        //cout << "resize image time cost: " << (1000000*(end_1.tv_sec - end_0.tv_sec) + end_1.tv_usec - end_0.tv_usec)/1000 << endl;

        vector<Blob<float>* > outputs = classifier_single.PredictBatch(images, 128.0, 128.0, 128.0);

        vector<vector<float> > areas;
        vector<vector<float> > ratios;
        //# 0
        //float car_08_area[3] = {400,800,1600};
        //float car_08_ratio[5] = {3,2,1,0.5,1.0/3};

        //# 1
        float car_16_area[7] = {1600,3200,6400,12800,25600,51200, 102400};
        float car_16_ratio[3] = {2,1,0.5};

        float add_1_area[3] = {1600,3200,6400};
        float add_1_ratio[2] = {1/3.0, 1/4.0};

        float add_2_area[1] = {800};
        float add_2_ratio[3] = {1/2.0, 1/3.0, 1/4.0};

        areas.push_back(vector<float> (car_16_area, car_16_area + 7));
        areas.push_back(vector<float> (add_1_area, add_1_area + 3));
        areas.push_back(vector<float> (add_2_area, add_2_area + 1));

        ratios.push_back(vector<float> (car_16_ratio, car_16_ratio + 3));
        ratios.push_back(vector<float> (add_1_ratio, add_1_ratio + 2));
        ratios.push_back(vector<float> (add_2_ratio, add_2_ratio + 3));

        
        float stride[] = {16};
        vector<Scalar> color;
        //color.push_back(Scalar(85,0,0));
        //color.push_back(Scalar(170,0,0));
        color.push_back(Scalar(255,0,0));

        vector<string> tags;
        //tags.push_back("car_08");
        tags.push_back("car_16");
        //tags.push_back("car_32");
        
        for(int cls_id = 0; cls_id < 1; cls_id ++) {
            vector<struct Bbox> result = get_final_bbox(images, outputs, cls_id, areas, ratios, global_confidence, enlarge_ratio, target_min_size, stride[cls_id]);
            for(int bbox_id = 0; bbox_id < result.size(); bbox_id ++) {
                rectangle(image, result[bbox_id].rect, color[cls_id]);
		char score[100];
		sprintf(score, "%.3f", result[bbox_id].confidence);
		fprintf(fid, "%d %d %d %d %f ", result[bbox_id].rect.x, result[bbox_id].rect.y, result[bbox_id].rect.x + result[bbox_id].rect.width, result[bbox_id].rect.y + result[bbox_id].rect.height, result[bbox_id].confidence);
                putText(image, tags[cls_id] + "_" + string(score), Point(result[bbox_id].rect.x, result[bbox_id].rect.y + 20), CV_FONT_HERSHEY_COMPLEX, 0.5,  Scalar(0,0,255));
            }
        }

        //struct timeval end_2;
        //gettimeofday(&end_2, NULL);
        //cout << "detection time cost: " << (1000000*(end_2.tv_sec - end_1.tv_sec) + end_2.tv_usec - end_1.tv_usec)/1000 << endl;
        //
        //struct timeval end_4;
        //gettimeofday(&end_4, NULL);
        //cout << "post detection [sort and nms] time cost: " << (1000000*(end_4.tv_sec - end_3.tv_sec) + end_4.tv_usec - end_3.tv_usec)/1000 << endl;

        struct timeval end;
        gettimeofday(&end, NULL);
        cout << "total time cost: " << (1000000*(end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec)/1000 << endl;
        fprintf(fid, "\n");

        cout << "rows: " << image.rows << " cols: " << image.cols << endl;
        //imwrite("debug.jpg", image);
        //imwrite("debug_resized.jpg", images[0]);
        imshow("debug.jpg", image);
        waitKey(-1);
  }
  fclose(fid);
  return 0;
}
