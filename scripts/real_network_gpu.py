
















# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Auto-scheduling a Neural Network for x86 CPU
============================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_, \
            `Chengfan Jia <https://github.com/jcf94/>`_

Auto-tuning for specific devices and workloads is critical for getting the
best performance. This is a tutorial on how to tune a whole neural
network for x86 CPU with the auto-scheduler.

To auto-tune a neural network, we partition the network into small subgraphs and 
tune them independently. Each subgraph is treated as one search task.
A task scheduler slices the time and dynamically allocates time resources to
these tasks. The task scheduler predicts the impact of each task on the end-to-end
execution time and prioritizes the one that can reduce the execution time the most.

For each subgraph, we use the compute declaration in :code:`tvm/python/topi` to
get the computational DAG in the tensor expression form.
We then use the auto-scheduler to construct a search space of this DAG and search
for good schedules (low-level optimizations).

Different from the template-based :ref:`autotvm <tutorials-autotvm-sec>` which relies on
manual templates to define the search space, the auto-scheduler does not require any
schedule templates. In other words, the auto-scheduler only uses the compute declarations
in :code:`tvm/python/topi` and does not use existing schedule templates.

Note that this tutorial will not run on Windows or recent versions of macOS. To
get it to run, you will need to wrap the body of this tutorial in a :code:`if
__name__ == "__main__":` block.
"""

import numpy as np
import os
import tvm
from tvm import relay, auto_scheduler
from tvm.relay import data_dep_optimization as ddo
import tvm.relay.testing
from tvm.contrib import graph_executor
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None, help='a chosen model, like resnet18_v2')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--n_trial', type=int, default=20000)
parser.add_argument("--tune", action='store_true')
parser.add_argument('--gpu_num', type=int, default=1)
parser.add_argument('--log',  type=str, default=None)
parser.add_argument('--mode',  type=str, default=None)
args = parser.parse_args()
print(args)


################################################################
# Get Network name
def get_name(input_name):
    global network
    global model
    global batch_size
    global log_file
    global layout
    global model_file
    batch_size = args.batch_size
    
    if input_name == 'resnet18_v1':
        network = "resnet-18"
        log_file = "%s-%s-B%d-%s.json" % (input_name, layout, batch_size, target.kind.name)
        model_file = "./models/%s-%s-B%d-%s.tar" % (input_name, layout, batch_size, target.kind.name)
    elif input_name == 'resnet34_v1':
        network = "resnet-34"
        log_file = "%s-%s-B%d-%s.json" % (input_name, layout, batch_size, target.kind.name)
        model_file = "./models/%s-%s-B%d-%s.tar" % (input_name, layout, batch_size, target.kind.name)
#    elif input_name == 'resnet50_v1':
#        network = "resnet-50"
#        log_file = "%s-%s-B%d-%s.json" % (input_name, layout, batch_size, target.kind.name)
#        model_file = "./models/%s-%s-B%d-%s.tar" % (input_name, layout, batch_size, target.kind.name)
    elif input_name == 'resnet101_v1':
        network = "resnet-101"
        log_file = "%s-%s-B%d-%s.json" % (input_name, layout, batch_size, target.kind.name)
        model_file = "./models/%s-%s-B%d-%s.tar" % (input_name, layout, batch_size, target.kind.name)
    elif input_name == 'resnet152_v1':
        network = "resnet-152"
        log_file = "%s-%s-B%d-%s.json" % (input_name, layout, batch_size, target.kind.name)
        model_file = "./models/%s-%s-B%d-%s.tar" % (input_name, layout, batch_size, target.kind.name)
    elif input_name == 'dcgan':
        network = "dcgan"  
        log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
        model_file = "./models/%s-%s-B%d-%s.tar" % (network, layout, batch_size, target.kind.name)
    elif input_name == 'frm':
        network = "frm"
        log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
        model_file = "./models/%s-%s-B%d-%s.tar" % (network, layout, batch_size, target.kind.name)
    elif input_name == 'graph_generation_opt':
        network = "graph_generation_opt"
        log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
        model_file = "./models/%s-%s-B%d-%s.tar" % (network, layout, batch_size, target.kind.name)    
    elif input_name == 'bert_base':
        network = "bert_base"
        log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
        model_file = "./models/%s-%s-B%d-%s.tar" % (network, layout, batch_size, target.kind.name)
    elif input_name == 'bert_large':
        network = "bert_large"
        log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
        model_file = "./models/%s-%s-B%d-%s.tar" % (network, layout, batch_size, target.kind.name)
    elif input_name == 'roberta_base':
        network = "roberta_base"
        log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
        model_file = "./models/%s-%s-B%d-%s.tar" % (network, layout, batch_size, target.kind.name)
    elif input_name == 'roberta_large':
        network = "roberta_large"
        log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
        model_file = "./models/%s-%s-B%d-%s.tar" % (network, layout, batch_size, target.kind.name)
    elif input_name == 'distilbert':
        network = "distilbert"
        log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
        model_file = "./models/%s-%s-B%d-%s.tar" % (network, layout, batch_size, target.kind.name)
    elif input_name == 'gpt2':
        network = "gpt2"
        log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
        model_file = "./models/%s-%s-B%d-%s.tar" % (network, layout, batch_size, target.kind.name)
    elif input_name == 'resnet3d-18':
        network = "resnet3d-18"
        layout = "NCDHW"
        log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
        model_file = "./models/%s-%s-B%d-%s.tar" % (network, layout, batch_size, target.kind.name)
    elif input_name == 'yolov3':
        network = "yolov3"
        log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
        model_file = "./models/%s-%s-B%d-%s.tar" % (network, layout, batch_size, target.kind.name)
    elif input_name == 'yolov3-tiny':
        network = "yolov3-tiny"
        log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
        model_file = "./models/%s-%s-B%d-%s.tar" % (network, layout, batch_size, target.kind.name)
    elif input_name == 'yolov2':
        network = "yolov2"
        log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
        model_file = "./models/%s-%s-B%d-%s.tar" % (network, layout, batch_size, target.kind.name)
    elif input_name == 'yolov2-tiny':
        network = "yolov2-tiny"
        log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
        model_file = "./models/%s-%s-B%d-%s.tar" % (network, layout, batch_size, target.kind.name)
    elif input_name == 'swin_transformer_small':
        network = "swin_transformer_small"
        log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
        model_file = "./models/%s-%s-B%d-%s.tar" % (network, layout, batch_size, target.kind.name)
    elif input_name == 'swin_transformer_base':
        network = "swin_transformer_base"
        log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
        model_file = "./models/%s-%s-B%d-%s.tar" % (network, layout, batch_size, target.kind.name)
    elif input_name == 'swin_transformer_large':
        network = "swin_transformer_large"
        log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
        model_file = "./models/%s-%s-B%d-%s.tar" % (network, layout, batch_size, target.kind.name)
    elif input_name == 'vit_base':
        network = "vit_base"
        log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
        model_file = "./models/%s-%s-B%d-%s.tar" % (network, layout, batch_size, target.kind.name)
    elif input_name == 'vit_huge':
        network = "vit_huge"
        log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
        model_file = "./models/%s-%s-B%d-%s.tar" % (network, layout, batch_size, target.kind.name)
    elif input_name == 'faster_rcnn':
        network = "faster_rcnn"
        log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
        model_file = "./models/%s-%s-B%d-%s.tar" % (network, layout, batch_size, target.kind.name)
    elif input_name == 'detr':
        network = "detr"
        log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
        model_file = "./models/%s-%s-B%d-%s.tar" % (network, layout, batch_size, target.kind.name)
    else:
        network = "mxnet"
        model = input_name
        log_file = "%s-%s-B%d-%s.json" % (model, layout, batch_size, target.kind.name)
        model_file = "./models/%s-%s-B%d-%s.tar" % (model, layout, batch_size, target.kind.name)





#################################################################
# Define a Network
# ----------------
# First, we need to define the network with relay frontend API.
# We can load some pre-defined network from :code:`tvm.relay.testing`.
# We can also load models from MXNet, ONNX, PyTorch, and TensorFlow
# (see :ref:`front end tutorials<tutorial-frontend>`).
#
# For convolutional neural networks, although auto-scheduler can work correctly
# with any layout, we found the best performance is typically achieved with NHWC layout.
# We also implemented more optimizations for NHWC layout with the auto-scheduler.
# So it is recommended to convert your models to NHWC layout to use the auto-scheduler.
# You can use :ref:`ConvertLayout <convert-layout-usage>` pass to do the layout conversion in TVM.

#import os
#num_threads = 1
#os.environ["TVM_NUM_THREADS"] = str(num_threads)

def get_network(name, batch_size, layout="NHWC", dtype="float32", use_sparse=False):
    """Get the symbol definition and random weight of a network"""
    # auto-scheduler prefers NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    elif layout == "NCDHW":
        image_shape = (3, 16, 112, 112)
    else:
        raise ValueError("Invalid layout: " + layout)
    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)
    global model
    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet_3d.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            #image_shape=input_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name =="dcgan":
        mod, params = relay.testing.dcgan.get_workload(
            batch_size=batch_size,
            dtype=dtype,
            layout=layout,
        )
    elif name == "mlp":
        mod, params = relay.testing.mlp.get_workload(
            batch_size=batch_size, dtype=dtype, image_shape=image_shape, num_classes=1000
        )
    elif name == "bert_base":
        # an example for mxnet model
        import mxnet as mx
        import gluonnlp as nlp

        model_name = 'bert_12_768_12'
        dataset = 'book_corpus_wiki_en_uncased'
        model, _ = nlp.model.get_model(
            name=model_name,
            dataset_name=dataset,
            pretrained=True,
            use_pooler=True,
            use_decoder=False,
            use_classifier=True)
        #model = nlp.model.BERTClassifier(bert, dropout=0.1, num_classes=2)
        #model.initialize(ctx=mx_ctx)
        #model.hybridize(static_alloc=True)    
        
        
        #assert layout == "NCHW"
        
        shape_dict = {
            'data0': (batch_size, seq_length),
            'data1': (batch_size, seq_length),
            'data2': (batch_size,)
        }
        
        #block = get_model(model, pretrained=True)
        mod, params = relay.frontend.from_mxnet(model, shape_dict)
        
        
        desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
        # Convert the layout to NHWC
        # RemoveUnunsedFunctions is used to clean up the graph.
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
    elif name == "bert_large":
        # an example for mxnet model
        import mxnet as mx
        import gluonnlp as nlp
        model_name = 'bert_24_1024_16'
        dataset = 'book_corpus_wiki_en_uncased'
        model, _ = nlp.model.get_model(
            name=model_name,
            dataset_name=dataset,
            pretrained=True,
            use_pooler=True,
            use_decoder=False,
            use_classifier=True)

        shape_dict = {
            'data0': (batch_size, seq_length),
            'data1': (batch_size, seq_length),
            'data2': (batch_size,)
        }
        #block = get_model(model, pretrained=True)
        mod, params = relay.frontend.from_mxnet(model, shape_dict)
        
        
        desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
        # Convert the layout to NHWC
        # RemoveUnunsedFunctions is used to clean up the graph.
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
    
    elif name == "roberta_base":
        # an example for mxnet model
        import mxnet as mx
        import gluonnlp as nlp
        model_name = 'roberta_12_768_12'
        dataset = 'openwebtext_ccnews_stories_books_cased'
        model, _ = nlp.model.get_model(
            name=model_name,
            dataset_name=dataset,
            pretrained=True,
            use_decoder=False)

        shape_dict = {
            'data0': (batch_size, seq_length)
        }
        #block = get_model(model, pretrained=True)
        mod, params = relay.frontend.from_mxnet(model, shape_dict)
        
        
        desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
        # Convert the layout to NHWC
        # RemoveUnunsedFunctions is used to clean up the graph.
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
    
    elif name == "roberta_large":
        # an example for mxnet model
        import mxnet as mx
        import gluonnlp as nlp
        model_name = 'roberta_24_1024_16'
        dataset = 'openwebtext_ccnews_stories_books_cased'
        model, _ = nlp.model.get_model(
            name=model_name,
            dataset_name=dataset,
            pretrained=True,
            use_pooler=True,
            use_decoder=False,
            use_classifier=True)

        shape_dict = {
            'data0': (batch_size, seq_length)
        }
        #block = get_model(model, pretrained=True)
        mod, params = relay.frontend.from_mxnet(model, shape_dict)
        
        
        desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
        # Convert the layout to NHWC
        # RemoveUnunsedFunctions is used to clean up the graph.
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)

    elif name == "distilbert":
        # an example for mxnet model
        import mxnet as mx
        import gluonnlp as nlp
        model_name = 'distilbert_6_768_12'
        dataset = 'distilbert_book_corpus_wiki_en_uncased'
        model, _ = nlp.model.get_model(
            name=model_name,
            dataset_name=dataset,
            pretrained=True)

        shape_dict = {
            'data0': (batch_size, seq_length),
            'data1': (batch_size,)
        }
        #block = get_model(model, pretrained=True)
        mod, params = relay.frontend.from_mxnet(model, shape_dict)
        
        
        desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
        # Convert the layout to NHWC
        # RemoveUnunsedFunctions is used to clean up the graph.
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
    
    elif name == "frm":
        input_name = "x"
        input_shape = (batch_size, 64, 7390)
        shape_dict = {input_name: input_shape}
        
        onnx_model = onnx.load("./onnx_models/final_frm-random-wei.onnx")
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        

        desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
        # Convert the layout to NHWC
        # RemoveUnunsedFunctions is used to clean up the graph.
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)         
    elif name == "graph_generation_opt":
        input_shape = (batch_size,33,512,512)
        shape_dict = {"kp_imgs":[batch_size,33,512,512], 'masks':[batch_size,3,512,512], 'targets':[batch_size,3,512,512]}
        
        onnx_model = onnx.load("/home/zsj/onnx_models/graph_generation_opt-constpad.onnx")
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        

        desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
        # Convert the layout to NHWC
        # RemoveUnunsedFunctions is used to clean up the graph.
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)         
    elif name == "gpt2":
        input_name = "input1"
        input_shape = (batch_size, seq_length, seq_length)
        shape_dict = {input_name: input_shape}
        
        onnx_model = onnx.load("./onnx_models/gpt2-10.onnx")
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        

        desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
        # Convert the layout to NHWC
        # RemoveUnunsedFunctions is used to clean up the graph.
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)         
    elif name == "swin_transformer_small":
        input_name = "input.1"
        input_shape = (batch_size, 3, 224, 224)
        shape_dict = {input_name: input_shape}
        
        onnx_model = onnx.load("./onnx_models/swin-transformer-small.onnx")
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

        desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
        # Convert the layout to NHWC
        # RemoveUnunsedFunctions is used to clean up the graph.
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod) 
    elif name == "swin_transformer_base":
        input_name = "data"
        input_shape = (batch_size, 3, 224, 224)
        shape_dict = {input_name: input_shape}
        
        onnx_model = onnx.load("./onnx_models/swin-transformer-base.onnx")
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

        desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
        # Convert the layout to NHWC
        # RemoveUnunsedFunctions is used to clean up the graph.
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod) 
    elif name == "swin_transformer_large":
        input_name = "data"
        input_shape = (batch_size, 3, 224, 224)
        shape_dict = {input_name: input_shape}
        
        onnx_model = onnx.load("./onnx_models/swin-transformer-large.onnx")
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

        desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
        # Convert the layout to NHWC
        # RemoveUnunsedFunctions is used to clean up the graph.
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod) 
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        #assert layout == "NCHW"
        #print(network, '+++++++++++', flush=True)
        #print(model, '-------------')
        block = get_model(model, pretrained=True)
        #print(network, '==============', flush=True)
        #print(model, '+++++++++')
        mod, params = relay.frontend.from_mxnet(block, shape={"data": (batch_size, 3, 224, 224)}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
        desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
        # Convert the layout to NHWC
        # RemoveUnunsedFunctions is used to clean up the graph.
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod) 
    elif name == "vit_base":
        import torch
        from vit_pytorch import ViT
        
        model = ViT(
            image_size = 224,
            patch_size = 16,
            num_classes = 1000,
            dim = 1024,
            depth = 12,
            heads = 12,
            mlp_dim = 3072,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        model = model.eval()
        #input_name = "data"
        #input_shape = (batch_size, 3, 224, 224)
        #shape_dict = {input_name: input_shape}
        input_shape = [1, 3, 224, 224]
        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, input_data).eval()
        #onnx_model = onnx.load("./onnx_models/ViT-B_16.onnx")
        #mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        input_name = "data"
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
        desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
        # Convert the layout to NHWC
        # RemoveUnunsedFunctions is used to clean up the graph.
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod) 
    elif name == "vit_huge":
        import torch
        from vit_pytorch import ViT
        #Base:768,12,12,3072
        #Large:1024,24,16,4096
        #Huge:1280,32,16,5120
        #patch:Lower is better min>14
        model = ViT(
            image_size = 224,
            patch_size = 16,
            num_classes = 1000,
            dim = 1280,
            depth = 32,
            heads = 16,
            mlp_dim = 5120,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        model = model.eval()

        input_shape = [1, 3, 224, 224]
        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, input_data).eval()
        input_name = "data"
        shape_list = [(input_name, input_shape)]
        
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
        desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
        # Convert the layout to NHWC
        # RemoveUnunsedFunctions is used to clean up the graph.
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
    elif name == "faster_rcnn":
        import gluoncv
        from gluoncv import model_zoo, data, utils
        net = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)
        input_shape = (1, 3, 224, 224)
        mod, params = relay.frontend.from_mxnet(net, {"data": input_shape})

        desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
        # Convert the layout to NHWC
        # RemoveUnunsedFunctions is used to clean up the graph.
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
    elif name == "detr":
        import torch
        import transformers

        configuration = transformers.DetrConfig(return_dict=False)
        model = transformers.DetrModel(configuration)
        model.eval()

        input_shape = [1, 3, 224, 224]
        input_data = torch.randn(input_shape)
        
        scripted_model = torch.jit.trace(model, input_data).eval()
        input_name = "data"
        shape_list = [(input_name, input_shape)]
        
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
        desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
        # Convert the layout to NHWC
        # RemoveUnunsedFunctions is used to clean up the graph.
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
    elif name == "yolov3":
        import gluoncv
        from gluoncv import model_zoo, data, utils
        net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)
        input_shape = (1, 3, 224, 224)
        mod, params = relay.frontend.from_mxnet(net, {"data": input_shape})

        desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
        # Convert the layout to NHWC
        # RemoveUnunsedFunctions is used to clean up the graph.
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
    elif "yolo" in name:
        from tvm.contrib.download import download_testdata
        from tvm.relay.testing.darknet import __darknetffi__
        import sys
        REPO_URL = "https://github.com/dmlc/web-data/blob/main/darknet/"
        if sys.platform in ["linux", "linux2"]:
            DARKNET_LIB = "libdarknet2.0.so"
            DARKNET_URL = REPO_URL + "lib/" + DARKNET_LIB + "?raw=true"
        elif sys.platform == "darwin":
            DARKNET_LIB = "libdarknet_mac2.0.so"
            DARKNET_URL = REPO_URL + "lib_osx/" + DARKNET_LIB + "?raw=true"
        else:
            err = "Darknet lib is not supported on {} platform".format(sys.platform)
            raise NotImplementedError(err)
        
        lib_path = download_testdata(DARKNET_URL, DARKNET_LIB, module="darknet")
        DARKNET_LIB = __darknetffi__.dlopen(lib_path)
        if name == "yolov3":
            CFG_NAME = name + ".cfg"
            WEIGHTS_NAME = name + ".weights"
            CFG_URL = REPO_URL + "cfg/" + CFG_NAME + "?raw=true"
            WEIGHTS_URL = "https://pjreddie.com/media/files/" + WEIGHTS_NAME
            cfg_path = download_testdata(CFG_URL, CFG_NAME, module="darknet")
            weights_path = download_testdata(WEIGHTS_URL, WEIGHTS_NAME, module="darknet")
            net = DARKNET_LIB.load_network(cfg_path.encode("utf-8"), weights_path.encode("utf-8"), 0)
            
            data = np.empty([batch_size, net.c, net.h, net.w], dtype)
            shape_dict = {"data": data.shape}
            input_shape = (batch_size, net.c, net.h, net.w)
            mod, params = relay.frontend.from_darknet(net, dtype=dtype, shape=data.shape)
        elif name == "yolov2":
            from tvm.contrib.download import download_testdata
            from tvm.relay.testing.darknet import __darknetffi__
            import sys
            CFG_NAME = name + ".cfg"
            WEIGHTS_NAME = name + ".weights"
            CFG_URL = REPO_URL + "cfg/" + CFG_NAME + "?raw=true"
            WEIGHTS_URL = "https://pjreddie.com/media/files/" + WEIGHTS_NAME
            cfg_path = download_testdata(CFG_URL, CFG_NAME, module="darknet")
            weights_path = download_testdata(WEIGHTS_URL, WEIGHTS_NAME, module="darknet")
            net = DARKNET_LIB.load_network(cfg_path.encode("utf-8"), weights_path.encode("utf-8"), 0)
            
            data = np.empty([batch_size, net.c, net.h, net.w], dtype)
            shape_dict = {"data": data.shape}
            input_shape = (batch_size, net.c, net.h, net.w)
            mod, params = relay.frontend.from_darknet(net, dtype=dtype, shape=data.shape)
        elif name == "yolov2-tiny":
            from tvm.contrib.download import download_testdata
            from tvm.relay.testing.darknet import __darknetffi__
            import sys
            CFG_NAME = name + ".cfg"
            WEIGHTS_NAME = name + ".weights"
            CFG_URL = REPO_URL + "cfg/" + CFG_NAME + "?raw=true"
            WEIGHTS_URL = "https://pjreddie.com/media/files/" + WEIGHTS_NAME
            cfg_path = download_testdata(CFG_URL, CFG_NAME, module="darknet")
            weights_path = download_testdata(WEIGHTS_URL, WEIGHTS_NAME, module="darknet")
            net = DARKNET_LIB.load_network(cfg_path.encode("utf-8"), weights_path.encode("utf-8"), 0)
            
            data = np.empty([batch_size, net.c, net.h, net.w], dtype)
            shape_dict = {"data": data.shape}
            input_shape = (batch_size, net.c, net.h, net.w)
            mod, params = relay.frontend.from_darknet(net, dtype=dtype, shape=data.shape)
        elif name == "yolov3-tiny":
            from tvm.contrib.download import download_testdata
            from tvm.relay.testing.darknet import __darknetffi__
            import sys
            CFG_NAME = name + ".cfg"
            WEIGHTS_NAME = name + ".weights"
            CFG_URL = REPO_URL + "cfg/" + CFG_NAME + "?raw=true"
            WEIGHTS_URL = "https://pjreddie.com/media/files/" + WEIGHTS_NAME
            cfg_path = download_testdata(CFG_URL, CFG_NAME, module="darknet")
            weights_path = download_testdata(WEIGHTS_URL, WEIGHTS_NAME, module="darknet")
            net = DARKNET_LIB.load_network(cfg_path.encode("utf-8"), weights_path.encode("utf-8"), 0)
            
            data = np.empty([batch_size, net.c, net.h, net.w], dtype)
            shape_dict = {"data": data.shape}
            input_shape = (batch_size, net.c, net.h, net.w)
            mod, params = relay.frontend.from_darknet(net, dtype=dtype, shape=data.shape)


        desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
        # Convert the layout to NHWC
        # RemoveUnunsedFunctions is used to clean up the graph.
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod) 
    else:
        raise ValueError("Network not found.")
        

    if use_sparse:
        from tvm.topi.sparse.utils import convert_model_dense_to_sparse

        mod, params = convert_model_dense_to_sparse(mod, params, random_params=True)

    return mod, params, input_shape, output_shape


# Define the neural network and compilation target.
# If the target machine supports avx512 instructions, replace the
# "llvm -mcpu=core-avx2" with "llvm -mcpu=skylake-avx512"
network = ""
model = ""
use_sparse = False
batch_size = args.batch_size
layout = "NHWC"
target = tvm.target.Target("cuda")
dtype = "float32"
log_file = "%s-%s-B%d-%s-%s-D%d.json" % (network, layout, batch_size, target.kind.name, args.mode, args.gpu_num)
model_file = "./models/%s-%s-B%d-%s-%s-D%d.tar" % (network, layout, batch_size, target.kind.name, args.mode, args.gpu_num)                                                                                          
#get_name()


#### Bert input ####
seq_length = 128
inputs = np.random.randint(0, 2000, size=(batch_size, seq_length)).astype(dtype)
token_types = np.random.uniform(size=(batch_size, seq_length)).astype(dtype)
valid_length = np.asarray([seq_length] * batch_size).astype(dtype)

#################################################################
# Extract Search Tasks
# --------------------
# Next, we extract the search tasks and their weights from a network.
# The weight of a task is the number of appearances of the task's subgraph
# in the whole network.
# By using the weight, we can approximate the end-to-end latency of the network
# as :code:`sum(latency[t] * weight[t])`, where :code:`latency[t]` is the
# latency of a task and :code:`weight[t]` is the weight of the task.
# The task scheduler will just optimize this objective.
last_network = 'str'
def inference(i, log_file, used_time, f_step_file):
    global network
    if (i != last_network):
    # Extract tasks from the network
        #print("Get model...")
        mod, params, input_shape, output_shape = get_network(
            network, batch_size, layout, dtype=dtype, use_sparse=use_sparse
        )
        print(network, '-----------')
        print(model, '-------------')
        #print("Extract tasks...")
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
        '''
        for idx, task in enumerate(tasks):
            print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
            print("task weight: %d" %(task_weights[idx]))
            print(task.compute_dag)
        '''
    print(network)
    print(model)
    #################################################################
    # Begin Tuning
    # ------------
    # Now, we set some options for tuning and launch the search tasks
    #
    # * :code:`num_measure_trials` is the number of measurement trials we can use during the tuning.
    #   You can set it to a small number (e.g., 200) for a fast demonstrative run.
    #   In practice, we recommend setting it around :code:`800 * len(tasks)`,
    #   which is typically enough for the search to converge.
    #   For example, there are 29 tasks in resnet-50, so we can set it as 20000.
    #   You can adjust this parameter according to your time budget.
    # * In addition, we use :code:`RecordToFile` to dump measurement records into a log file,
    #   The measurement records can be used to query the history best, resume the search,
    #   and do more analyses later.
    # * see :any:`auto_scheduler.TuningOptions`,
    #   :any:`auto_scheduler.LocalRunner` for more parameters.
    #


    def run_tuning(device_number):
        '''
        print("Begin tuning...")
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(
            repeat=1, min_repeat_ms=300, timeout=10)

        tuner = auto_scheduler.TaskScheduler(
            tasks, task_weights)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=900 * (len(tasks) + 1),  # change this to 20000 to achieve the best performance
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )

        tuner.tune(tune_option)
        '''
        assert device_number > 0
        print("Begin tuning...")

        ctx_list = []
        print("Set environment variable")
        os.environ['TVM_DEVICE_NUMBER'] = str(device_number)
        for i in range(device_number):
            ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)
            ctx_list.append(ctx)
            if i != 0:
                os.environ[f'TVM_RPC_KEY-{i}'] = ctx.key
                os.environ[f'TVM_RPC_HOST-{i}'] = ctx.host
                os.environ[f'TVM_RPC_PORT-{i}'] = str(ctx.port)


        tuner = auto_scheduler.TaskScheduler(
                tasks, task_weights, load_log_file=log_file)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=900 * (len(tasks) + 1),  # change this to 20000 to achieve the best performance
            runner=ctx_list[0].runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )
        tuner.tune(tune_option)

        for ctx in ctx_list:
            del ctx

    # We do not run the tuning in our webpage server since it takes too long.
    # Uncomment the following line to run it by yourself.

    import time
    tuning_begin = time.time()
    if args.tune:
        run_tuning(args.gpu_num)
    tuning_time = time.time() - tuning_begin
    
    ######################################################################
    # .. note:: Explain the printed information during tuning
    #
    #   During the tuning, a lot of information will be printed on the console.
    #   They are used for debugging purposes. The most important info is the output
    #   of the task scheduler. The following table is a sample output.
    #
    #   .. code-block:: c
    #
    #     ----------------------------------------------------------------------
    #     ------------------------------  [ Task Scheduler ]
    #     ----------------------------------------------------------------------
    #     |  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
    #     -------------------------------------------------
    #     |    0 |        0.010 |           0.40 |     64 |
    #     |    1 |        0.087 |          47.19 |     64 |
    #     |    2 |        0.008 |          -0.00 |     64 |
    #     |    3 |        0.177 |         582.07 |     64 |
    #     |    4 |        0.268 |         862.37 |    256 |
    #     |    5 |        0.166 |         621.13 |    128 |
    #     |    6 |        0.170 |         605.10 |    128 |
    #     |    7 |        0.128 |         403.20 |     64 |
    #     |    8 |        0.189 |         545.71 |     64 |
    #     |    9 |        0.231 |        1001.01 |    448 |
    #     |   10 |        0.155 |         664.80 |    256 |
    #     |   11 |        0.155 |         662.86 |    256 |
    #     |   12 |        0.119 |         434.08 |     64 |
    #     |   13 |        0.199 |         522.13 |     64 |
    #     |   14 |        0.235 |         986.56 |    320 |
    #     |   15 |        0.149 |         689.13 |    128 |
    #     |   16 |        0.155 |         664.80 |    192 |
    #     |   17 |        0.151 |         340.64 |     64 |
    #     |   18 |        0.176 |         597.55 |    128 |
    #     |   19 |        0.220 |        1054.37 |    192 |
    #     |   20 |        0.150 |         686.01 |    128 |
    #     |   21 |        0.159 |         650.88 |    128 |
    #     |   22 |        0.073 |         358.19 |     64 |
    #     |   23 |        0.031 |          70.63 |     64 |
    #     |   24 |        0.251 |         947.73 |    128 |
    #     |   25 |        0.157 |         652.47 |    128 |
    #     |   26 |        0.215 |         954.84 |    128 |
    #     |   27 |        0.237 |         868.92 |    128 |
    #     |   28 |        0.266 |         774.06 |    128 |
    #     -------------------------------------------------
    #     Estimated total latency: 10.016 ms      Trials: 3992    Used time : 1131 s      Next ID: 15
    #
    #   This table lists the latency and (estimated) speed of all tasks.
    #   It also lists the allocation of measurement trials for all tasks.
    #   The last line prints the total weighted latency of these tasks,
    #   which can be a rough estimation of the end-to-end execution time
    #   of the network.
    #   The last line also prints the total number of measurement trials,
    #   total time spent on auto-tuning and the id of the next task to tune.
    #
    #   There will also be some "tvm::Error"s errors, because the
    #   auto-scheduler will try some invalid schedules.
    #   You can safely ignore them if the tuning can continue, because these
    #   errors are isolated from the main process.
    #

    ######################################################################
    # .. note:: Terminate the tuning earlier
    #
    #   You can terminate the tuning earlier by forcibly killing this process.
    #   As long as you get at least one valid schedule for each task in the log file,
    #   you should be able to do the compilation (the secion below).
    #


    #################################################################
    # Compile and Evaluate
    # --------------------
    # After auto-tuning, we can compile the network with the best schedules we found.
    # All measurement records are dumped into the log file during auto-tuning,
    # so we can read the log file and load the best schedules.

    # Compile with the history best
    #print("Compile...")
    #with auto_scheduler.ApplyHistoryBest(log_file):
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=target, params=params)
            #lib.export_library(model_file)
            
    # Create graph executor
    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))

    if network == "bert_base":
        module.set_input(data0=inputs, data1=token_types, data2=valid_length)
    elif network == "bert_large":
        module.set_input(data0=inputs, data1=token_types, data2=valid_length)
    elif network == "roberta_base":
        module.set_input(data0=inputs)    
    elif network == "roberta_large":
        module.set_input(data0=inputs)     
    elif network == "distilbert":
        module.set_input(data0=inputs, data1=valid_length)     
    elif network == "frm":   
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input("x", data_tvm)
    elif network == "graph_generation_opt":
        input_shape = (batch_size,33,512,512)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input("kp_imgs", data_tvm)
        
        input_shape = (batch_size,3,512,512)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input("masks", data_tvm)
        
        input_shape = (batch_size,3,512,512)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input("targets", data_tvm)                                                             
    elif network == "gpt2":   
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype("int64"))
        module.set_input("input1", data_tvm)
    elif network == "dcgan":
            input_shape = (batch_size,100)
            data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
            module.set_input("data", data_tvm)
            #out = module.get_output(0, tvm.nd.empty(output_shape, dtype))    
    elif network == "swin_transformer_small":
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input("input.1", data_tvm)
    elif network == "vit_base":
        input_shape = (batch_size,3,224,224)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input("data", data_tvm)
    elif network == "vit_huge":
        input_shape = (batch_size,3,224,224)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input("data", data_tvm)
    else:
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input("data", data_tvm)

    # Evaluate
    print("Evaluate inference time cost...")
    ftimer = module.module.time_evaluator("run", dev, number=10, repeat=3, min_repeat_ms=800)
    prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
    #print("Mean inference time (std dev): %.3f ms (%.3f ms) %d s" % (np.mean(prof_res), np.std(prof_res), used_time))
    print("Mean inference time (std dev): %.3f ms (%.3f ms)" % (np.mean(prof_res), np.std(prof_res)))
    f_step_file.write(str("Mean inference time (std dev): %.3f ms (%.3f ms) %d s" % (np.mean(prof_res), np.std(prof_res), used_time))+'\n')
    f_step_file.flush()
    def fmt(t: float) -> str:
        "Format time in second to HH:mm:ss. "
        return f'{int(t/60/60):0>2d}:{int(t/60%60):0>2d}:{int(t%60%60):0>2d}'

    #print(f'network: {network}, tuning time: {fmt(tuning_time)}, {tuning_time}')

    #################################################################
    # Other Tips
    # ----------
    # 1. During the tuning, the auto-scheduler needs to compile many programs and
    #    extract feature from them. This part is CPU-intensive,
    #    so a high-performance CPU with many cores is recommended for faster search.
    # 2. You can use :code:`python3 -m tvm.auto_scheduler.measure_record --mode distill -i log.json`
    #    to distill the large log file and only save the best useful records.
    # 3. You can resume a search from the previous log file. You just need to
    #    add a new argument :code:`load_log_file` when creating the task scheduler
    #    in function :code:`run_tuning`. Say,
    #    :code:`tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=log_file)`
    # 4. If you have multiple target CPUs, you can use all of them for measurements to
    #    parallelize the measurements. Check this :ref:`section <tutorials-autotvm-scale-up-rpc-tracker>`
    #    to learn how to use the RPC Tracker and RPC Server.
    #    To use the RPC Tracker in auto-scheduler, replace the runner in :code:`TuningOptions`
    #    with :any:`auto_scheduler.RPCRunner`.



def get_x_y_list(filename,index,method):
    data = [[], [], []]
    f = open(filename, 'r')

    symbol = dict()
    symbol['latency'] = ['Trials: ', '	']
    symbol['time'] = ['Used time : ', ' s']
    symbol['ntrial'] = ['Get ', ' programs to measure']

    for line in f:
        if line.find(symbol['latency'][0]) != -1 and line.find(symbol['time'][0]) != -1:
            strings = symbol['latency'][0]
            first = line.find(strings) + len(strings)
            strings = symbol['latency'][1]
            end = line.find(strings)
            temp = line[first:].split("\t")[0]
            try:
                temp = int(temp)
            except:
                temp = None

            #if method == "family":
            #    data[0].append(temp+pretrain_trial[index])
            #else:
            data[0].append(temp)
            

            line = line[end+len(strings):]
            strings = symbol['time'][0]
            first = line.find(strings) + len(strings)
            strings = symbol['time'][1]
            end = line.find(strings)
            temp = line[first:end]
            temp = int(temp)
            #temp = None if temp == 0 else temp
            #if method == "family":
            #    data[1].append(temp-pretrain_time[index])
            #else:
            data[1].append(temp)
        if line.find(symbol['ntrial'][1]) != -1:
            strings = symbol['ntrial'][0]
            first = line.find(strings) + len(strings)
            strings = symbol['ntrial'][1]
            end = line.find(strings)
            temp = line[first:end]
            #print(temp)
            try:
                temp = int(temp)
            except:
                temp = None

            #if method == "family":
            #    data[0].append(temp+pretrain_trial[index])
            #else:
            data[2].append(temp)
            

            
    f.close()

    return data

models = [
    'resnet50_v1',
    'resnet152_v2',
    'mobilenet0.5',
    'mobilenetv2_0.5',
    'vit_huge',
    'bert_large', 
    'roberta_large',
    'gpt2'
    ]
pretrain_step = [9]
#pretrain_step = [25,32,22,34,13,9,11,10]
targets = ['cuda']
platform = ['NVIDIA_V100']
methods = ['family', 'ansor']
#get_name('resnet50_v1')
#inference('resnet50_v1','file_name','data[1][pretrain_step[index]+step]','f_step_file')
def get_filename(model, method, platform):
    if args.gpu_num > 1:
        return "[{}]_[{}]_{}_B1_D{}.output".format(model, platform ,method, args.gpu_num)
    return "[{}]_[{}]_{}_B1.output".format(model, platform ,method)

def get_jsonname(model, targets, methods):
    return "{}-NHWC-B1-{}-{}-D{}.json".format(model, targets, methods, args.gpu_num)




#data_gpu_family = [get_x_y_list(get_filename(i, methods[0], targets[0])) for i in models]
#data_gpu_default = [get_x_y_list(get_filename(i, methods[0], platform[0])) for i in models]




def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'w')
    for i in data:
        s=i
          #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()

file_idx = 0
'''       
for met in methods:   
    for index,i in enumerate(models):
        print (i)
        f_step_file= open("/home/zsj/log/{}_{}_{}.step".format(i, platform[0],met),'w')
        data = get_x_y_list(get_filename(i, met, platform[0]),index,met)
        max = data[0][-1]-data[0][0]
        #print (len(data[0]),len(data[1]))
        step = 500
        step_gate = step
        c_step = 0
        if met == "family":
            data_step = 1
        else:
            data_step = 0
        if (i == 'resnet50_v1'):
            data_step = 29
        elif (i == 'resnet152_v2'):
            data_step = 32
        f = open(get_jsonname(i, targets[0],met),"r")
        json_log = []
        for line in f:
            #print (line)
            if c_step==data[0][data_step]:
                file_name = "/home/zsj/log/json/{}-{}-{}.json".format(i, met,str(file_idx))
                text_save(file_name,json_log)
                get_name(i)
                
                try:
                    inference(i,file_name,data[1][data_step],f_step_file)
                except:
                    print("Error")
                if (i == 'resnet50_v1'):
                    last_network = ""
                else:
                    last_network = i
                file_idx += 1
                data_step += 1
            if(data_step == len(data[0])):
                break
            json_log.append(line)
            c_step += 1
        f_step_file.close()
        '''
for index,i in enumerate(models):
    for met in methods:
        print (i)
        f_step_file= open("./log_data/realbench_log/{}_{}_{}_B1_D{}.step".format(i, platform[0], met, args.gpu_num),'w')
        data = get_x_y_list(get_filename(i, met, platform[0]),index,met)
        max = data[0][-1]-data[0][0]
        #print (len(data[0]))
        '''
        l_step=0
        print(len(data[2]))
        for step in data[0][pretrain_step[index]:]:
            if l_step>=step:
                print("error")
            l_step=step
        '''
        if(len(data[0])!=len(data[2])):
            print("length unmatch!")
        c_step = data[0][pretrain_step[index]]
        json_log = []
        f = open(get_jsonname(i, targets[0], met),"r")
        for line in f:
            json_log.append(line)
        f.close()
        #print(len(json_log))
        for step,data_step in enumerate(data[2][pretrain_step[index]:]):
            #print(data_step)

            file_name = "./json/{}-{}-{}.json".format(i, met,str(file_idx))
            text_save(file_name,json_log[:c_step])
            get_name(i)
            log_file = "%s-%s-B%d-%s-%s-D%d.json" % (network, layout, batch_size, target.kind.name, met, args.gpu_num)
            model_file = "./models/%s-%s-B%d-%s-%s-D%d.tar" % (network, layout, batch_size, target.kind.name, met, args.gpu_num)
            c_step = c_step + data_step
            inference(i,file_name,data[1][pretrain_step[index]+step],f_step_file)
            file_idx += 1
        f_step_file.close() 