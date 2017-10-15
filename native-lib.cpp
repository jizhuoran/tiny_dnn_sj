#include <jni.h>
#include <string>
#include <algorithm>
#include <vector>
#include <memory>
#include <streambuf>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <tiny_dnn/util/image.h>
#include <tiny_dnn/tiny_dnn.h>
#include "third_party/CLCudaAPI/clpp11.h"
#include "clManager.hpp"
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

#define APPNAME "TINYDNNANDROID"


#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO,APPNAME,__VA_ARGS__)

// rescale output to 0-100
template <typename Activation>
double rescale(double x) {
    Activation a;
    return 100.0 * (x - a.scale().first) / (a.scale().second - a.scale().first);
}

struct membuf : std::streambuf
{
    membuf(char* begin, char* end) {
        this->setg(begin, begin, end);
    }
};


network<sequential> nn;
void convert_image(const std::string& imagefilename,
                   double minv,
                   double maxv,
                   int w,
                   int h,
                   vec_t& data) {

    image<> img(imagefilename, image_type::grayscale);
    image<> resized = resize_image(img, w, h);

    // mnist dataset is "white on black", so negate required
    std::transform(resized.begin(), resized.end(), std::back_inserter(data),
                   [=](uint8_t c) { return (255 - c) * (maxv - minv) / 255.0 + minv; });
}


float ret[10];
static double now_ms(void) {
    struct timespec res;
    clock_gettime(CLOCK_REALTIME, &res);
    return 1000.0 * res.tv_sec + (double) res.tv_nsec / 1e6;
}


tiny_dnn::device_t device_type(size_t *platform, size_t *device) {
    // check which platforms are available
    auto platforms = CLCudaAPI::GetAllPlatforms();

    // if no platforms - return -1
    if (platforms.size() == 0) {
        return tiny_dnn::device_t::NONE;
    }
    std::array<std::string, 2> devices_order        = {"GPU", "CPU"};
    std::map<std::string, tiny_dnn::device_t> devices_t_order = {
            std::make_pair("GPU", tiny_dnn::device_t::GPU), std::make_pair("CPU", tiny_dnn::device_t::CPU)};
    for (auto d_type : devices_order)
        for (auto p = platforms.begin(); p != platforms.end(); ++p)
            for (size_t d = 0; d < p->NumDevices(); ++d) {
                auto dev = CLCudaAPI::Device(*p, d);
                if (dev.Type() == d_type) {
                    *platform = p - platforms.begin();
                    *device   = d;
                    return devices_t_order[d_type];
                }
            }
    // no CPUs or GPUs
    return tiny_dnn::device_t::NONE;
}

static void construct_net(tiny_dnn::network<tiny_dnn::sequential> &nn,
                          tiny_dnn::core::backend_t backend_type) {
// connection table [Y.Lecun, 1998 Table.1]
#define O true
#define X false
    // clang-format off
    static const bool tbl[] = {
            O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
            O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
            O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
            X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
            X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
            X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
    };
// clang-format on
#undef O
#undef X

    // construct nets
    //
    // C : convolution
    // S : sub-sampling
    // F : fully connected
    // clang-format off
    using fc = tiny_dnn::layers::fc;
    using conv = tiny_dnn::layers::conv;
    using ave_pool = tiny_dnn::layers::ave_pool;
    using tanh = tiny_dnn::activation::tanh;

    using tiny_dnn::core::connection_table;
    using padding = tiny_dnn::padding;

    nn << conv(32, 32, 5, 1, 6,   // C1, 1@32x32-in, 6@28x28-out
               padding::valid, true, 1, 1, backend_type)
       << tanh()
       << ave_pool(28, 28, 6, 2)   // S2, 6@28x28-in, 6@14x14-out
       << tanh()
       << conv(14, 14, 5, 6, 16,   // C3, 6@14x14-in, 16@10x10-out
               connection_table(tbl, 6, 16),
               padding::valid, true, 1, 1, backend_type)
       << tanh()
       << ave_pool(10, 10, 16, 2)  // S4, 16@10x10-in, 16@5x5-out
       << tanh()
       << conv(5, 5, 5, 16, 120,   // C5, 16@5x5-in, 120@1x1-out
               padding::valid, true, 1, 1, backend_type)
       << tanh()
       << fc(120, 10, true, backend_type)  // F6, 120-in, 10-out
       << tanh();
}

size_t cl_platform = 0, cl_device = 0;
tiny_dnn::device_t device = device_type(&cl_platform, &cl_device);
float* recognize(const std::string& dictionary, const std::string& src_filename) {

    //__android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "111111");

    // convert imagefile to vec_t
   tiny_dnn::models::alexnet nn1;
    //std::string s;

    tiny_dnn::Device my_gpu_device(device, cl_platform, cl_device);

//    __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "begin libdnn");
//    __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "begin libdnn");
//    __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "begin libdnn");
//    __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "begin libdnn");

    tiny_dnn::network<tiny_dnn::sequential> nn_mnist;
    construct_net(nn_mnist, tiny_dnn::core::backend_t::internal);
    nn_mnist.load("/storage/emulated/0/data/LeNet-model",  content_type::weights, file_format::json);
    //tiny_dnn::models::alexnet nn_mnist;
    // change all layers at once
    //nn_mnist.weight_init(tiny_dnn::weight_init::constant(2.0));
   // nn_mnist.bias_init(tiny_dnn::weight_init::constant(2.0));
    //nn_mnist.init_weight();

/*
    for (int i = 0; i < nn_mnist.depth(); i++) {
        std::cout << "#layer:" << i << "\n";
        std::cout << "layer type:" << nn_mnist[i]->layer_type() << "\n";
        std::cout << "input:" << nn_mnist[i]->in_size() << "(" << nn_mnist[i]->in_shape() << ")\n";
        std::cout << "output:" << nn_mnist[i]->out_size() << "(" << nn_mnist[i]->out_shape() << ")\n";
        if(nn_mnist[i]->layer_type()=="conv") {
            nn_mnist[i]->set_backend_type(tiny_dnn::core::backend_t::libdnn);
            std::cout << "yes this layer is ok";
            my_gpu_device.registerOp(*nn_mnist[i]);

        }
    }
*/


    for (int i = 0; i < nn1.depth(); i++) {
        std::cout << "#layer:" << i << "\n";
        std::cout << "layer type:" << nn1[i]->layer_type() << "\n";
        std::cout << "input:" << nn1[i]->in_size() << "(" << nn1[i]->in_shape() << ")\n";
        std::cout << "output:" << nn1[i]->out_size() << "(" << nn1[i]->out_shape() << ")\n";
        if(nn1[i]->layer_type()=="conv") {
            nn1[i]->set_backend_type(tiny_dnn::core::backend_t::opencl);
            std::cout << "yes this layer is ok";
            my_gpu_device.registerOp(*nn1[i]);

        }
    }







    // generate random variables

    tiny_dnn::vec_t in_dnn(224 * 224 * 3);
    tiny_dnn::vec_t in_dnn1(32 * 32 * 1);

    tiny_dnn::uniform_rand(in_dnn.begin(), in_dnn.end(), 0, 1);

/*
    // predic
    int slice_all[48][3] = {{1, 1, 1}, {1, 2, 1}, {1, 4, 1}, {1, 8, 1}, {1, 16, 1}, {1, 32, 1}, {2, 1, 1}, {2, 2, 1}, {2, 4, 1}, {2, 8, 1}, {2, 16, 1}, {2, 32, 1}, {4, 1, 1}, {4, 2, 1}, {4, 4, 1}, {4, 8, 1}, {4, 16, 1}, {4, 32, 1}, {8, 1, 1}, {8, 2, 1}, {8, 4, 1}, {8, 8, 1}, {8, 16, 1}, {8, 32, 1}, {16, 1, 1}, {16, 2, 1}, {16, 4, 1}, {16, 8, 1}, {16, 16, 1}, {16, 32, 1}, {32, 1, 1}, {32, 2, 1}, {32, 4, 1}, {32, 8, 1}, {32, 16, 1}, {32, 32, 1}, {64, 1, 1}, {64, 2, 1}, {64, 4, 1}, {64, 8, 1}, {64, 16, 1}, {64, 32, 1}, {128, 1, 1}, {128, 2, 1}, {128, 4, 1}, {128, 8, 1}, {128, 16, 1}, {128, 32, 1}};
    WZCL::stretch_factor[0] = 1;
    WZCL::stretch_factor[1] = 1;
    WZCL::stretch_factor[2] = 1;

    for(int i = 0; i < 8; ++i) {
        WZCL::slice_factor[0] = slice_all[i+15][0];
        WZCL::slice_factor[1] = slice_all[i+15][1];
        WZCL::slice_factor[2] = slice_all[i+15][2];

        __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "slice_factor 0: %d slice_factor 1: %d slice_factor 2: %d", WZCL::slice_factor[0], WZCL::slice_factor[1], WZCL::slice_factor[2]);

        //__android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "begin benchmark1111");
*/

//    WZCL::stretch_factor[0] = 16;
//    WZCL::stretch_factor[1] = 16;
//    WZCL::stretch_factor[2] = 1;
//
//    WZCL::slice_factor[0] = 1;
//    WZCL::slice_factor[1] = 1;
//    WZCL::slice_factor[2] = 1;
//
//
//    double time_dnn = 0.0;
//    double alltime = 0.0; ;

   /* for(int j = 0; j < 1; ++j) {
        tiny_dnn::uniform_rand(in_dnn1.begin(), in_dnn1.end(), 0, 1);
        time_dnn = now_ms();
        auto res_dnn = nn1.predict(in_dnn1);
        alltime += (now_ms() - time_dnn);
    }*/

    vec_t data;
    convert_image("/storage/emulated/0/data/digit.png", -1.0, 1.0, 32, 32, data);

    double before_time = now_ms();
    //auto res = nn_mnist.predict(data);
    auto res = nn1.predict(in_dnn);




    for (int i = 0; i < 10; i++) {
        __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "predicting results:%d %lf\n", i,
                            rescale<tiny_dnn::tanh_layer>(res[i]));

        // ret[i] = rescale<tan_h>(res[i]);
        ret[i] = rescale<tiny_dnn::tanh_layer>(res[i]);
    }

    __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "The time used is !!! %lf\n", now_ms() - before_time);
    return ret;
    // save outputs of each layer
}



static double benchmark(){
    models::alexnet nn;

    // change all layers at once
    nn.weight_init(weight_init::constant(2.0));
    nn.bias_init(weight_init::constant(2.0));
    nn.init_weight();

    vec_t in(224 * 224 * 3);

    // generate random variables
    uniform_rand(in.begin(), in.end(), 0, 1);

    timer t; // start the timer

    // predict
    auto res = nn.predict(in);

    double elapsed_ms = t.elapsed();
    t.stop();

    __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "AlexNet benchmark elapsed time(s): %lf\n", elapsed_ms );

    return elapsed_ms;
}


float* recognize(const std::string& dictionary, jfloat* digitsdata) {




    // convert imagefile to vec_t
    vec_t data;
    for(int i = 0; i<32*32;++i){
        data.push_back(digitsdata[i]);
    }

    __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "before do the native predict");
    // recognize
    auto res = nn.predict(data);
    __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "after do the native predict");

    for (int i = 0; i < 10; i++) {
        __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "predicting results:%d %lf\n", i,
                            rescale<tiny_dnn::tanh_layer>(res[i]));

        ret[i] = rescale<tiny_dnn::tanh_layer>(res[i]);
    }
    return ret;

}




/*
static void construct_net(tiny_dnn::network<tiny_dnn::sequential> &nn,
                          tiny_dnn::core::backend_t backend_type) {
// connection table [Y.Lecun, 1998 Table.1]
#define O true
#define X false
    // clang-format off
    static const bool tbl[] = {
            O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
            O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
            O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
            X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
            X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
            X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
    };
// clang-format on
#undef O
#undef X

    // construct nets
    //
    // C : convolution
    // S : sub-sampling
    // F : fully connected
    // clang-format off
    using fc = tiny_dnn::layers::fc;
    using conv = tiny_dnn::layers::conv;
    using ave_pool = tiny_dnn::layers::ave_pool;
    using tanh = tiny_dnn::activation::tanh;

    using tiny_dnn::core::connection_table;
    using padding = tiny_dnn::padding;

    nn << conv(32, 32, 5, 1, 6,   // C1, 1@32x32-in, 6@28x28-out
               padding::valid, true, 1, 1, backend_type)
       << tanh()
       << ave_pool(28, 28, 6, 2)   // S2, 6@28x28-in, 6@14x14-out
       << tanh()
       << conv(14, 14, 5, 6, 16,   // C3, 6@14x14-in, 16@10x10-out
               connection_table(tbl, 6, 16),
               padding::valid, true, 1, 1, backend_type)
       << tanh()
       << ave_pool(10, 10, 16, 2)  // S4, 16@10x10-in, 16@5x5-out
       << tanh()
       << conv(5, 5, 5, 16, 120,   // C5, 16@5x5-in, 120@1x1-out
               padding::valid, true, 1, 1, backend_type)
       << tanh()
       << fc(120, 10, true, backend_type)  // F6, 120-in, 10-out
       << tanh();
}
*/

static void train_lenet(const std::string &data_dir_path,
                        double learning_rate,
                        const int n_train_epochs,
                        const int n_minibatch,
                        tiny_dnn::core::backend_t backend_type) {
    // specify loss-function and learning strategy
    tiny_dnn::network<tiny_dnn::sequential> nn;
    tiny_dnn::adagrad optimizer;

    construct_net(nn, backend_type);

    std::cout << "load models..." << std::endl;
    __android_log_print(ANDROID_LOG_INFO,APPNAME,"load models");

    // load MNIST dataset
    std::vector<tiny_dnn::label_t> train_labels, test_labels;
    std::vector<tiny_dnn::vec_t> train_images, test_images;

    tiny_dnn::parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte",
                                 &train_labels);
    tiny_dnn::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte",
                                 &train_images, -1.0, 1.0, 2, 2);
    tiny_dnn::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte",
                                 &test_labels);
    tiny_dnn::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte",
                                 &test_images, -1.0, 1.0, 2, 2);


    __android_log_print(ANDROID_LOG_INFO,APPNAME,"start training");
    std::cout << "start training" << std::endl;

    tiny_dnn::progress_display disp(train_images.size());
    tiny_dnn::timer t;

    optimizer.alpha *=
            std::min(tiny_dnn::float_t(4),
                     static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate));

    int epoch = 1;
    // create callback
    auto on_enumerate_epoch = [&]() {
        std::cout << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
                  << t.elapsed() << "s elapsed." << std::endl;
        ++epoch;
        __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "before");

        tiny_dnn::result res = nn.test(test_images, test_labels);
        __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "after");

        std::cout << res.num_success << "/" << res.num_total << std::endl;

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_minibatch = [&]() { disp += n_minibatch; };

    // training
    nn.train<tiny_dnn::mse>(optimizer, train_images, train_labels, n_minibatch,
                            n_train_epochs, on_enumerate_minibatch,
                            on_enumerate_epoch);

    std::cout << "end training." << std::endl;

    // test and show results
    nn.test(test_images, test_labels).print_detail(std::cout);
    // save network model & trained weights
    nn.save("/storage/emulated/0/data/LeNet-model", content_type::weights_and_model, file_format::json);
}

static tiny_dnn::core::backend_t parse_backend_name(const std::string &name) {
    const std::array<const std::string, 5> names = {{
                                                            "internal", "nnpack", "libdnn", "avx", "opencl",
                                                    }};
    for (size_t i = 0; i < names.size(); ++i) {
        if (name.compare(names[i]) == 0) {
            return static_cast<tiny_dnn::core::backend_t>(i);
        }
    }
    return tiny_dnn::core::default_engine();
}

static void usage(const char *argv0) {
    std::cout << "Usage: " << argv0 << " --data_path path_to_dataset_folder"
              << " --learning_rate 1"
              << " --epochs 30"
              << " --minibatch_size 16"
              << " --backend_type internal" << std::endl;
}

/*
namespace models {

// Based on:
// https://github.com/DeepMark/deepmark/blob/master/torch/image%2Bvideo/alexnet.lua
    class alexnet : public tiny_dnn::network<tiny_dnn::sequential> {
    public:
        explicit alexnet(const std::string &name = "")
                : tiny_dnn::network<tiny_dnn::sequential>(name) {
            // todo: (karandesai) shift this to tiny_dnn::activation
            using relu     = tiny_dnn::activation::relu;
            using conv     = tiny_dnn::layers::conv;
            using max_pool = tiny_dnn::layers::max_pool;
            *this << conv(224, 224, 11, 11, 1, 64, padding::valid, true, 4, 4);
            *this << relu(54, 54, 64);
            *this << max_pool(54, 54, 64, 2);
            *this << conv(27, 27, 5, 5, 64, 192, padding::valid, true, 1, 1);
            *this << relu(23, 23, 192);
            *this << max_pool(23, 23, 192, 1);
            *this << conv(23, 23, 3, 3, 192, 384, padding::valid, true, 1, 1);
            *this << relu(21, 21, 384);
            *this << conv(21, 21, 3, 3, 384, 256, padding::valid, true, 1, 1);
            *this << relu(19, 19, 256);
            *this << conv(19, 19, 3, 3, 256, 256, padding::valid, true, 1, 1);
            *this << relu(17, 17, 256);
            *this << max_pool(17, 17, 256, 1);
        }
    };
}*/

static JavaVM* g_JavaVM = NULL;


jobject getInstance(JNIEnv *env, jclass obj_class)
{
    jmethodID  c_id = env->GetMethodID(obj_class, "<init>", "()V");
    jobject obj = env->NewObject(obj_class, c_id);
    return obj;
}


void test_for() {

    JNIEnv *env;
    __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "come here");
    if(g_JavaVM == NULL) {
        __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "it's null");
    }


    g_JavaVM->GetEnv((void**)&env, JNI_VERSION_1_6);

    __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "come here11111");

    //jstring jstr1 = env->NewStringUTF(s.c_str());

    jclass clazz = env->FindClass("com/myndk/getBattery");
    __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "come 22222222");

    if (clazz == NULL) {
        __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "why it is null????");

    }

    jobject obj;
    obj = getInstance(env, clazz);

    jmethodID mid = env->GetMethodID(clazz, "get_battery_level", "(Ljava/lang/String;)V");


    if (mid == NULL) {
        __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "why mid it is null????");

    }


    __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "come 3333333");

    jstring jstrMSG = env->NewStringUTF("tmp");

    env->CallVoidMethod(obj, mid, jstrMSG);
}









extern "C" {

void  Java_com_tinydnn_android_MainActivity_loadModel(JNIEnv* env,jclass tis
        ,jobject assetManager,jstring filename)
{


    float* a = recognize("a","b");
/*

    // change all layers at once
    nn1.weight_init(tiny_dnn::weight_init::constant(2.0));
    nn1.bias_init(tiny_dnn::weight_init::constant(2.0));
    nn1.init_weight();



    __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "begin opencl");
    __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "begin opencl");
    __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "begin opencl");
    __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "begin opencl");

    for (int i = 0; i < nn1.depth(); i++) {
        std::cout << "#layer:" << i << "\n";
        std::cout << "layer type:" << nn1[i]->layer_type() << "\n";
        std::cout << "input:" << nn1[i]->in_size() << "(" << nn1[i]->in_shape() << ")\n";
        std::cout << "output:" << nn1[i]->out_size() << "(" << nn1[i]->out_shape() << ")\n";
        if(nn1[i]->layer_type()=="conv") {
            nn1[i]->set_backend_type(tiny_dnn::core::backend_t::opencl);
            std::cout << "yes this layer is ok";
            my_gpu_device.registerOp(*nn1[i]);

        }
    }


    tiny_dnn::vec_t in(224 * 224 * 3);

    // generate random variables
    tiny_dnn::uniform_rand(in.begin(), in.end(), 0, 1);


    // predict

    __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "begin benchmark");
    double time1 = now_ms();
    auto res = nn1.predict(in);
    double time = now_ms() - time1;
    __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "the time used is %s", std::to_string(time).c_str());
    __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "after benchmark");

*/






        //__android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "begin benchmark22222");

        //__android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "the time used is %s", std::to_string(time_dnn).c_str());
        //__android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "after benchmark22222");






        //__android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "after benchmark1111");

 //   }








    /*

    LOGI("ReadAssets");
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    if(mgr==NULL)
    {
        LOGI(" %s","AAssetManager==NULL");
        return ;
    }

    jboolean iscopy;
    const char *mfile = env->GetStringUTFChars(filename, &iscopy);
    AAsset* asset = AAssetManager_open(mgr, mfile,AASSET_MODE_UNKNOWN);
    env->ReleaseStringUTFChars(filename, mfile);
    if(asset==NULL)
    {
        LOGI(" %s","asset==NULL");
        return ;
    }

    //construct_net(nn);
    off_t bufferSize = AAsset_getLength(asset);
    LOGI("file size         : %d\n",bufferSize);
    char *buffer=(char *)malloc(bufferSize+1);
    buffer[bufferSize]=0;
    int numBytesRead = AAsset_read(asset, buffer, bufferSize);

   // membuf sbuf(buffer, buffer + bufferSize);
   // std::ifstream in("/storage/emulated/0/data/LeNet-model");
    //construct_net(nn, tiny_dnn::core::default_engine());

    //in >> nn;
    nn.load("/storage/emulated/0/data/cifar-weights", content_type::weights_and_model, file_format::json);
    //nn.load("/storage/emulated/0/data/LeNet-model",  content_type::weights_and_model, file_format::binary);
    //nn.load("/sdcard/LeNet-model");
   // LOGI(": %s",buffer);
    free(buffer);
    AAsset_close(asset);



    double learning_rate                   = 1;
    int epochs                             = 1;
    std::string data_path                  = "/storage/emulated/0/data/mnist";
    int minibatch_size                     = 16;
    tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();


    std::cout << "Running with the following parameters:" << std::endl
              << "Data path: " << data_path << std::endl
              << "Learning rate: " << learning_rate << std::endl
              << "Minibatch size: " << minibatch_size << std::endl
              << "Number of epochs: " << epochs << std::endl
              << "Backend type: " << backend_type << std::endl
              << std::endl;
    try {
        train_lenet(data_path, learning_rate, epochs, minibatch_size, backend_type);
    } catch (tiny_dnn::nn_error &err) {
        std::cerr << "Exception: " << err.what() << std::endl;
    }
*/

}

jstring
Java_com_tinydnn_android_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    double elapsed = benchmark();
    std::string hello = "AlexNet benchmark elapsed time(s):" + std::to_string(elapsed);
    return env->NewStringUTF(hello.c_str());
}
jfloatArray JNICALL Java_com_tinydnn_android_MainActivity_recognize
        (JNIEnv *env, jobject obj, jfloatArray fltarray1)
{

    //std::ifstream in("/storage/emulated/0/data/LeNet-model");
   // tiny_dnn::network<tiny_dnn::sequential> nn1;
    //in >> nn;
    //nn1.load("/storage/emulated/0/data/LeNet-model");
   // nn.load("/storage/emulated/0/data/LeNet-model",  content_type::weights_and_model, file_format::binary);

    jfloatArray result;
    result = env->NewFloatArray(10);
    if (result == NULL) {
        return NULL; /* out of memory error thrown */
    }

    jfloat array1[10];
    jfloat* flt1 = env->GetFloatArrayElements(fltarray1,0);

    //float *current = recognize("/sdcard/LeNet-model", flt1);
    __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "before recognizing bitmap in sdcard");

    float * current = recognize("/sdcard/LeNet-model", "/sdcard/data/cat.jpg");
    __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "after recognizing bitmap in sdcard");

    for(int i = 0;i<10;++i) {
        array1[i] = current[i];
        __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "ret:%d %lf\n", i,
                            array1[i]);

    }

    env->ReleaseFloatArrayElements(fltarray1, flt1, 0);
    env->SetFloatArrayRegion(result, 0, 10, array1);
    return result;

}

}