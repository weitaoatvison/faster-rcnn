
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
#include <iostream>

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
  void get_detection(vector<Blob<float>* >& outputs, vector<struct Bbox> &final_vbbox);

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

  //CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
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

    float mean_[3] = {102.9801, 115.9465, 122.7717};
    for(int k = 0; k < sample.channels(); k++) {
        for(int i = 0; i < sample.rows; i++) {
            for(int j = 0; j < sample.cols; j++) {
               input_data[cnt] = (float(sample.at<uchar>(i,j*3+k))-mean_[k]);
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

  string layer_name_rois_ = "rois";
  string layer_name_score_ = "cls_prob";
  string layer_name_bbox_ = "bbox_pred";

  Blob<float>* output_rois = net_->blob_by_name(layer_name_rois_).get();
  Blob<float>* output_score = net_->blob_by_name(layer_name_score_).get();
  Blob<float>* output_bbox = net_->blob_by_name(layer_name_bbox_).get();
  outputs.push_back(output_rois);
  outputs.push_back(output_score);
  outputs.push_back(output_bbox);

  //for(int i = 0; i < net_->num_outputs(); i++) {
  //  Blob<float>* output_layer = net_->output_blobs()[i];
  //  outputs.push_back(output_layer);
  //}
  return outputs;
}

struct Bbox {
    float confidence;
    Rect rect;
    Rect gt;
    bool deleted;
    int cls_id;

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

void bbox_transform_inv_clip(Blob<float>* roi, Blob<float>* cls, Blob<float>* reg, Blob<float>* im_info_layer, vector<struct Bbox> &vbbox)
{
    const float* roi_cpu = roi->cpu_data();
    const float* cls_cpu = cls->cpu_data();
    const float* reg_cpu = reg->cpu_data();
    const float* im_info = im_info_layer->cpu_data();

    vbbox.resize(roi->shape()[0]);
    for (int i = 0; i < roi->shape()[0]; i++)
    {

        if(i==0) {
            cout << 0 << " " << roi_cpu[0] << endl;
            cout << 1 << " " << roi_cpu[1] << endl;
            cout << 2 << " " << roi_cpu[2] << endl;
            cout << 3 << " " << roi_cpu[3] << endl;
            cout << 4 << " " << roi_cpu[4] << endl;
        }
        int cls_id = 0;
        float prob = 0;
        for(int li = 0; li <5; li++) {
            if(prob < cls_cpu[5*i + li]) {
                prob = cls_cpu[5*i+li];
                cls_id = li;
            }
        }
        if(cls_id == 0 || prob < 0.5) {
            continue;
        }
        int width, height, ctr_x, ctr_y;
        float dx, dy, dw, dh, pred_ctr_x, pred_ctr_y, pred_w, pred_h;

        const float* cur_reg = reg_cpu + i*20 + 4*cls_id;
        const float* cur_roi = roi_cpu + i*5 + 1;
        width  = cur_roi[2] - cur_roi[0] + 1.0;
        height = cur_roi[3] - cur_roi[1] + 1.0;
        ctr_x = cur_roi[0] + 0.5 * width;
        ctr_y = cur_roi[1] + 0.5 * height;
        dx = cur_reg[0];
        dy = cur_reg[1];
        dw = cur_reg[2];
        dh = cur_reg[3];
        pred_ctr_x = dx * width + ctr_x;
        pred_ctr_y = dy * height + ctr_y;
        pred_w = exp(dw) * width;
        pred_h = exp(dh) * height;

        // clip_boxes
        //int x1 = std::max(std::min(float(pred_ctr_x - 0.5 * pred_w), float(im_info[1] -1)),float(0));
        //int y1 = std::max(std::min(float(pred_ctr_y - 0.5 * pred_h), float(im_info[0] -1)),float(0));
        //int x2 = std::max(std::min(float(pred_ctr_x + 0.5 * pred_w), float(im_info[1] -1)),float(0));
        //int y2 = std::max(std::min(float(pred_ctr_y + 0.5 * pred_h), float(im_info[0] -1)),float(0));
        struct Bbox &bbox = vbbox[i];
        
        if(cls_id >= 1) {
            bbox.confidence = prob;
            bbox.cls_id = cls_id;
        } else {
            bbox.confidence = 1.0 - prob;
        }
        bbox.rect = Rect(pred_ctr_x - 0.5 * pred_w, pred_ctr_y - 0.5 * pred_h, pred_w, pred_h) & Rect(0, 0, im_info[1] - 1, im_info[0] - 1);
        bbox.deleted = false;
    }
}


void CaffeClassifier::get_detection(vector<Blob<float>* >& outputs, vector<struct Bbox> &final_vbbox)
{
    Blob<float>* roi = outputs[0];
    Blob<float>* cls = outputs[1];
    Blob<float>* reg = outputs[2];

    assert(roi->shape()[1] == 5);
    assert(cls->num() == reg->num());
    assert(cls->channels() == 5);
    assert(reg->channels() == 20);
    assert(cls->height() == reg->height());
    assert(cls->width() == reg->width());

    //for(int i = 0; i < 10; i++) {
    //    std::cout << i << " " << roi[i] << std::endl;
    //}


    vector<struct Bbox> vbbox;

    Blob<float>* im_info_layer = net_->input_blobs()[1];
    const float* im_info = im_info_layer->cpu_data();
    bbox_transform_inv_clip(roi, cls, reg, im_info_layer, vbbox);
    float resize_ratio = im_info[2];

    if (vbbox.size()!=0)
    {

        sort(vbbox.begin(), vbbox.end(), mycmp);
        int max_per_img_ = 100;
        vbbox.resize(min(static_cast<size_t>(max_per_img_), vbbox.size()));
        nms(vbbox, 0.2);
    }

    final_vbbox.resize(0);
    for(size_t i = 0; i < vbbox.size(); i++)
    {

        if(!vbbox[i].deleted)
        {
            struct Bbox box = vbbox[i];
            float x = box.rect.x / resize_ratio;
            float y = box.rect.y / resize_ratio;
            float w = box.rect.width / resize_ratio;
            float h = box.rect.height / resize_ratio;
            box.rect.x = x;
            box.rect.y = y;
            box.rect.width = w;
            box.rect.height = h;
            if (vbbox[i].confidence > 0.7) //conf_thres_)
            {
                final_vbbox.push_back(box);
            }
        }
    }
}




int main(int argc, char** argv) {

  google::InitGoogleLogging(argv[0]);

  // caffe variables
  string prefix = ""; 
  string model_file   = "/mnt/data1/zdb/work/caffe_for_frcnn/zuozhen/py-faster-rcnn/models/GoogleNet_inception5/faster_rcnn_end2end/test.prototxt";  //prefix + "/mnt/ssd/DeepV/faster_rcnn/detector/test.prototxt"; //  rcnn_googlenet.prototxt
  string trained_file = "/mnt/data1/zdb/work/caffe_for_frcnn/zuozhen/py-faster-rcnn/output/GoogleNet/deepv_car_and_person/resized/frcnn_train/googlenet_faster_rcnn_iter_350000.caffemodel"; //prefix + "/mnt/ssd/DeepV/faster_rcnn/detector/googlenet_faster_rcnn_iter_48000.caffemodel"; // rcnn_googlenet.caffemodel
  
  //////////////////
  // string image_list = "/mnt/ssd/DeepV/roi/nightfront_v1.3/truckData_aug.txt";


  CaffeClassifier  classifier_single(model_file, trained_file, true, 1);
  string image_list = "test.list"; // "/mnt/ssd/DeepV/roi/nightfront_v1.3/truckData_aug.txt"; //
  FILE *fcin  = fopen(image_list.c_str(),"r");
  if(!fcin) {
    cout << "can not open filelist" << endl;
  }
  char image_filename[200];

  cout << "single mode" << endl;

  int tot_cnt = 0;

  float global_confidence = 0.9; //atof(argv[7]);
  while(fscanf(fcin, "%s", image_filename)!=EOF) {
    struct timeval start;
    gettimeofday(&start, NULL);
    tot_cnt += 1;
    //if(tot_cnt < 28380) continue;
	//fprintf(fid, "%s ", image_filename);
    cout << "filename " << string(image_filename) << endl;
    vector<Mat> images;
    Mat image = imread(image_filename, -1);
    if (image.empty()) {
        cout << "Wrong Image" << endl;
        continue;
    }
    
     
    //cv::cvtColor(image, image, CV_BGR2RGB);
    //resize(image, image, Size(600, int(image.rows*1.0/image.cols*600)));
    //struct timeval end_0;
    //gettimeofday(&end_0, NULL);
    //cout << "readling image time cost: " << (1000000*(end_0.tv_sec - start.tv_sec) + end_0.tv_usec - start.tv_usec)/1000 << endl;

    Mat img;

    float border_ratio = 0.00;

    img = image.clone();

    int max_size = max(img.rows, img.cols);
    int min_size = min(img.rows, img.cols);
    float target_min_size = 450.0;
    float target_max_size = 1000.0;
    float enlarge_ratio = target_min_size / min_size;

    if(max_size * enlarge_ratio > target_max_size) {
        enlarge_ratio = target_max_size / max_size;
    }

    int target_row = img.rows * enlarge_ratio;
    int target_col = img.cols * enlarge_ratio;

    resize(img, img, Size(target_col, target_row));

    images.push_back(img);
    vector<Blob<float>* > outputs = classifier_single.PredictBatch(images, 128.0, 128.0, 128.0);

    vector<struct Bbox> result;
    classifier_single.get_detection(outputs, result);

	struct timeval end;
	gettimeofday(&end, NULL);
	double fps = (double(end.tv_sec - start.tv_sec) + double(end.tv_usec - start.tv_usec) / 1.0e6);
    cout << "timecost = "<< fps << endl;

    float det_thresh = 0.7;

    vector<Scalar> colors;
    colors.push_back(Scalar(0,0,0));
    colors.push_back(Scalar(255,0,0));
    colors.push_back(Scalar(0,255,0));
    colors.push_back(Scalar(0,0,255));
    colors.push_back(Scalar(255,255,0));

    vector<string> class_names;
    class_names.push_back("bg");
    class_names.push_back("car");
    class_names.push_back("person");
    class_names.push_back("bike");
    class_names.push_back("tricycle");
	for(size_t bbox_id = 0; bbox_id < result.size(); bbox_id ++) {
        int cls_id = result[bbox_id].cls_id;
		rectangle(image, result[bbox_id].rect, colors[cls_id],3);
		char str_prob[100];
		sprintf(str_prob,"%s_%.3f", class_names[cls_id].c_str(), result[bbox_id].confidence);
		string info(str_prob);
		putText(image, info, Point(result[bbox_id].rect.x, result[bbox_id].rect.y), CV_FONT_HERSHEY_SIMPLEX, 0.5,  Scalar(0,0,255));
	}

	cv::imshow("Cam",image);
    cv::waitKey(-1);
	//if (cv::waitKey(1)>=0)
	  //stop = true;
  }
  return 0;
}
