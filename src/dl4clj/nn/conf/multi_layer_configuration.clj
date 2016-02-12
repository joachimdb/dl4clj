(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/MultiLayerConfiguration.html"}
  dl4clj.nn.conf.multi-layer-configuration
  (:require [dl4clj.nn.conf.backprop-type :as backprop-type])
  (:import [org.deeplearning4j.nn.conf MultiLayerConfiguration MultiLayerConfiguration$Builder]))

(defn builder 
  ([]
   (builder (MultiLayerConfiguration$Builder.)))
  ([^MultiLayerConfiguration$Builder builder {:keys [backprop ;; Whether to do back prop or not (boolean)
                                                     backprop-type ;; (one of (backprop-type/values)) 
                                                     cnn-input-size ;; CNN input size, in order of [height,width,depth] (int-array) 
                                                     confs ;; java.util.List<NeuralNetConfiguration>	
                                                     input-pre-processors ;; ({integer,InputPreProcessor})	
                                                     pretrain ;; Whether to do pre train or not (boolean)
                                                     redistribute-params ;; Whether to redistribute parameters as a view or not (boolean)
                                                     t-bptt-backward-length ;; When doing truncated BPTT: how many steps of backward should we do?
                                                     ;; Only applicable when doing backpropType(BackpropType.TruncatedBPTT)
                                                     ;; This is the k2 parameter on pg23 of http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf(int) 
                                                     t-bptt-forward-length ;; When doing truncated BPTT: how many steps of forward pass should we do before doing (truncated) backprop? (int)
                                                     ;; Only applicable when doing backpropType(BackpropType.TruncatedBPTT)
                                                     ;; Typically tBPTTForwardLength parameter is same as the the tBPTTBackwardLength parameter, but may be larger than it in some circumstances (but never smaller)
                                                     ;; Ideally your training data time series length should be divisible by this This is the k1 parameter on pg23 of http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf
                                                     ]
                                              :or {}
                                              :as opts}]
   (when backprop
     (.backprop builder backprop))
   (when backprop-type
     (.backpropType builder (backprop-type/value-of backprop-type)))
   (when cnn-input-size
     (.cnnInputSize builder (int-array cnn-input-size)))
   (when confs
     (.confs builder confs))
   (when input-pre-processors
     (.inputPreProcessors builder input-pre-processors))
   (when pretrain
     (.pretrain builder pretrain))
   (when redistribute-params
     (.redistributeParams builder redistribute-params))
   (when t-bptt-backward-length
     (.tBPTTBackwardLength builder t-bptt-backward-length))
   (when t-bptt-forward-length
     (.tBPTTForwardLength builder t-bptt-forward-length))
   builder))


(comment
  
  (builder {:backprop true
            :backprop-type :truncated-bptt
            :cnn-input-size [3 4 5]
            :conf [(dl4clj.nn.conf.neural-net-configuration/neural-net-configuration
                    {:layer (dl4clj.nn.conf.layers.graves-lstm/graves-lstm-layer)})]
            :input-pre-processors {0 (dl4clj.nn.conf.preprocessor.cnn-to-rnn-pre-processor/cnn-to-rnn-pre-processor 3 4 5)}
            :pretrain true
            :redistribute-params true
            :tbptt-backward-length 3
            :tbptt-forward-length 3
            })
  
)
