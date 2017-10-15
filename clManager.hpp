//
//  clManager.hpp
//  clKernelManager
//
//  Created by NATURE on 1/8/2017.
//  Copyright © 2017 NATURE. All rights reserved.
//

#ifndef clManager_hpp
#define clManager_hpp
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include<CL/cl.h>
#endif
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string.h>
#include <string>
#include <regex>
#include <queue>
#include <stdarg.h>
#include <stdio.h>

using std::string;
using std::vector;
using std::cout;
using std::endl;
//record how many places to change

namespace WZCL{
    /*macros*/
#define ckE(n) if(n!=0)printf("error happend in line %d with error [%d] in file [%s]\n",__LINE__-1,n,__FILE__);
    
    
    enum KERNEL_EVENT_TYPE{DO_NOT_GENERATE_EVENTS, GENERATE_EVENTS};
    
    /*slice & stretch*/
    extern int slice_factor[3];
    extern int stretch_factor[3];
    
    /*cl_event list related, record all the events*/
    /*
     events:        e0*  e1*   e2*   e3* ... en*
     numOfEvents      n0    n1    n2     n3   ...  nn*
     
     */
    extern std::vector<std::vector<cl_event>>eventList[2];
    void getTimeFromEventList(int training_or_testing);
    extern int netPointer;   // to specify whether now is a testing net or training net: 0-> training net, 1-> testing net
    void SetTrainingNet();  //set netPointer = 0
    void SetTestingNet();   // set netPointer = 1
    
    
    //each layer is correspond to an event vector
    class eventLayer{
        long event_layer_counter; //the id of this layer among the layers with events
        std::vector<cl_event> event_in_this_layer; //each layer is associated with an event vector
        
    public:
        //The events in this layer has the same number of slice_factor
        eventLayer(){
            printf("called event layer\n");
            eventList[netPointer].push_back(event_in_this_layer);
            event_layer_counter = eventList[netPointer].size()-1;
        }
        //get event of this layer
        std::vector<cl_event> getEventsThistLayer(){
            return event_in_this_layer;
        }
        //get events of next layer
        std::vector<cl_event> getEventsNextLayer(){
            if(event_layer_counter==(eventList[netPointer].size()-1)){
                return {};
            }
            
            return eventList[netPointer].at(event_layer_counter+1);
        }
        size_t getNumEventsNextLayer(){
            if(event_layer_counter==(eventList[netPointer].size()-1)){
                return {};
            }
            return eventList[netPointer].at(event_layer_counter+1).size();
        }
        void clearEventsInThisLayer(){
            //std::cout<<"Going to clear"<<event_in_this_layer.size()<<" events in this layer\n";
            event_in_this_layer.clear();
        }
        void addEventToThisLayer(cl_event event){
            event_in_this_layer.push_back(event);
        }
        ~eventLayer(){
            
        }
    };
    
    
    
    
    
    
    
    /*options*/
    extern int PARALLEL_EXECUTIN_KERNEL;
    extern int PRINT_LIB_CONV_INFO;
    extern int KERNEL_FOR_LOOP_LEVEL;
    
    
    
    
    
    /*opencl related variables*/
    /*classes*/
    //NDRange(cl_command_queue que,cl_kernel ker, int d, size_t* off, size_t* gsize, size_t* lsize, int nw);
    //cl_event executeKernel()
    class NDRange;
    
    
    
    
    /*Kernel related*/
    /*strings*/
    extern int placesToModify;
    extern string fromKeywords[12];
    extern string toKeywords[12];
    extern string fromParameterKeywords[2];
    extern string toParameterKeywords[2];
    extern string toFrontBody;
    extern string toEndBody;
    /*functions*/
    //given kernel name, return its position in the string
    size_t findKernelPosition(const string source, string kernel_name);
    // find function scop
    void findFuntionScop(size_t& start, size_t& end, string source, size_t startKernelPosition);
    // find function scop and replace the tail
    void findFuntionScopAndAddTail(size_t& start, size_t& end, string& source, size_t startKernelPosition);
    //replace keywords such as global_id(0)...
    void replaceKeywords(int& counter, string& source, string keyword, string new_keyword,size_t start, size_t end);
    //add  parameters and for loop
    void replaceBody(size_t kernel_name_position, string& source, int& counter);
    //replace kernels according to kernel name
    void replaceKernel(string& source, const string kernelname, int& modified_counter);
    //main entrance for kernel string replacement
    void modifyKernel(string& kernel_source, const char* kernel_name);
    const char* updateKernel(const char* kernel_source, const char* kernel_name);
    //You should specify kernel names you want to change
    //modify kernel entrance
    cl_program JppClCreateProgramWithSource(cl_context context, cl_uint count,const char **strings, const size_t *lengths,cl_int *errcode_ret, int numOfKernelsToModify, const char** kernelnames);
    
    
    
    
    
    
    
    /* Host related to lauch kernels*/
    //Host side, push kernels to kernel_queues
    void reshape_kernel(size_t* globalsize, size_t* localsize, int* slice, int* stretch ,cl_kernel& kernel, cl_command_queue queue,int arg_base, std::queue<NDRange*>& launch_queue, eventLayer* event_layer,KERNEL_EVENT_TYPE kernel_event_type);
    void reshape_kernel_no_events(size_t* globalsize, size_t* localsize, int* slice, int* stretch ,cl_kernel& kernel, cl_command_queue queue,int arg_base, std::queue<NDRange*>& launch_queue);
    
    
    
    
    // Set kernel args
    //obbey the order: cl_kernel& kernel, int n_buffer, (cl_mem* mem1, cl_uint index){1,n}
    void JppSetMemKernelArgs(cl_kernel& kernel, int n_mem, ... );
    //order: cl_kernel& kernel, int n_buffer, (size_t data_size, void* scalar_data, cl_uint index){1,n}
    void JppSetScalarKernelArgs(cl_kernel& kernel, int n_scalar_data, ... );
    //Launch kernels
    void kernel_execution();
    
    
    /*opencl kernels*/
    //Those kernels in different queues are associated with different opencl command queues
    extern std::queue <NDRange*> kernel_queue_1;
    extern std::queue <NDRange*> kernel_queue_2;
    //Prepare for OpenCL, context and device is not initiated
    void get_context_device_from_queue(cl_command_queue queue, cl_context* context, cl_device_id* device);
    extern std::vector<cl_command_queue> cl_concurrent_command_queue;
    //the maximum number of cl_command_queues to create
    extern int MAX_CL_COMMAND_QUEUE;
    void addAConcurrentCLCommandQueueFromOriginalQueue(cl_command_queue queue);
    void addAConcurrentCLCommandQueueFromContext(cl_context context,
                             cl_device_id device,
                             cl_command_queue_properties properties);
    cl_command_queue getConcurrentCLCommandQueue(size_t queue_index);
    
    
}
#endif /* clManager_hpp */
