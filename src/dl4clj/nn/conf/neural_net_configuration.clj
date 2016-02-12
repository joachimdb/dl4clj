(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/NeuralNetConfiguration.html"}
  dl4clj.nn.conf.neural-net-configuration
  (:require [dl4clj.nn.conf.multi-layer-configuration :as ml-configuration]
            [dl4clj.nn.conf.gradient-normalization :as gradient-normalization]
            [dl4clj.nn.conf.updater :as updater]
            [dl4clj.nn.api.optimization-algorithm :as opt]
            [dl4clj.nn.weights.weight-init :as weight-init]
            [clojure.data.json :as json]
            [dl4clj.utils :refer (camel-to-dashed)])
  (:import [org.deeplearning4j.nn.conf NeuralNetConfiguration NeuralNetConfiguration$Builder NeuralNetConfiguration$ListBuilder]
           ;; [org.nd4j.linalg.factory Nd4j]
           ))

(defn list-builder [^NeuralNetConfiguration$Builder b layers opts]
  (println "-----> list-builder!" (count layers))
  (let [builder ^NeuralNetConfiguration$ListBuilder (ml-configuration/builder b opts)]
    (doseq [[idx ^Layer layer] layers]
      (println "adding layer" idx layer "(type" (type layer) ")")
      (.layer builder idx layer))
    builder))

(defn builder [{:keys [activation ;; Activation function / neuron non-linearity Typical values include "relu" (rectified linear), "tanh", "sigmoid", "softmax", "hardtanh", "leakyrelu", "maxout", "softsign", "softplus"
                       adam-mean-decay ;; Mean decay rate for Adam updater (double)
                       adam-var-decay  ;; Variance decay rate for Adam updater (double)
                       bias-init       ;; (double)
                       dist            ;; Distribution to sample initial weights from (Distribution)
                       drop-out        ;; (double)
                       gradient-normalization ;; Gradient normalization strategy (one of (dl4clj.nn.conf.gradient-normalization/values))
                       gradient-normalization-threshold ;; Threshold for gradient normalization, only used for :clip-l2-per-layer, :clip-l2-per-param-type 
                       ;; and clip-element-wise-absolute-value: L2 threshold for first two types of clipping, or absolute 
                       ;; value threshold for last type of clipping.

                       num-iterations      ;; Number of optimization iterations (int)
                       l1                  ;; L1 regularization coefficient (double)
                       l2                  ;; L2 regularization coefficient (double)
                       layer               ;; (Layer)
                       learning-rate       ;; (double)
                       learning-rate-after ;; Learning rate schedule ({integer,double})
                       learning-rate-score-based-decay-rate ;; Rate to decrease learningRate by when the score stops improving (double)
                       max-num-line-search-iterations       ;; (integer)
                       mini-batch ;; Process input as minibatch vs full dataset (boolean)
                       minimize ;; Objective function to minimize or maximize cost function. Default=true (boolean)
                       momentum ;; Momentum rate (double)
                       momentum-after    ;; Momentum schedule ({integer,double})
                       optimization-algo ;; (one of (dl4clj.nn.api.optimization-algorithm/values))
                       regularization ;; Whether to use regularization (l1, l2, dropout, etc.) (boolean)
                       rho            ;; Ada delta coefficient (double)
                       rms-decay      ;; Decay rate for RMSProp (double)
                       schedules ;; Whether to use schedules :learning-rate-after and :momentum-after (boolean)
                       seed      ;; Random number generator seed (int or long)
                       step-function ;; Step function to apply for back track line search (org.deeplearning4j.optimize.api.StepFunction)
                       updater       ;; Gradient updater (one of (dl4clj.nn.conf.updater/values))
                       use-drop-connect ;; Use drop connect: multiply the coefficients by a binomial sampling wrt the dropout probability (boolean)
                       weight-init ;; Weight initialization scheme (one of (dl4clj.nn.weights.weight-init/values))
                       ;; multi-layer parameters
                       list ;; Number of layers not including input (int)
                       layers ;; {idx,Layer}
                       backprop ;; Whether to do back prop or not (boolean)
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
  (let [b (NeuralNetConfiguration$Builder.)]
    (when activation
      (.activation b activation))
    (when adam-mean-decay
      (.adamMeanDecay b adam-mean-decay))
    (when adam-var-decay
      (.adamVarDecay b adam-var-decay))
    (when bias-init
      (.biasInit b bias-init))
    (when dist
      (.dist b dist))
    (when drop-out
      (.dropOut b drop-out))
    (when gradient-normalization
      (.gradientNormalization b gradient-normalization))
    (when gradient-normalization-threshold
      (.gradientNormalizationThreshold b gradient-normalization-threshold))
    (when num-iterations
      (.iterations b num-iterations))
    (when l1
      (.l1 b l1))
    (when l2
      (.l2 b l2))
    (when layer
      (.layer b layer))
    (when learning-rate
      (.learningRate b learning-rate))
    (when learning-rate-after
      (.learningRateAfter b learning-rate-after))
    (when learning-rate-score-based-decay-rate
      (.learningRateScoreBasedDecayRate b learning-rate-score-based-decay-rate))
    (when max-num-line-search-iterations
      (.maxNumLineSearchIterations b max-num-line-search-iterations))
    (when mini-batch
      (.miniBatch b mini-batch))
    (when minimize
      (.minimize b minimize))
    (when momentum
      (.momentum b momentum))
    (when momentum-after
      (.momentumAfter b momentum-after))
    (when optimization-algo
      (.optimizationAlgo b (opt/value-of optimization-algo)))
    (when regularization
      (.regularization b regularization))
    (when rho
      (.rho b rho))  
    (when rms-decay
      (.rmsDecay b rms-decay))
    (when schedules
      (.schedules b schedules))
    ;; (when seed
    ;;   (.seed b seed))
    (when step-function
      (.stepFunction b step-function))
    (when updater
      (.updater b updater))
    (when use-drop-connect
      (.useDropConnect b use-drop-connect))
    (when weight-init
      (.weightInit b weight-init))
    (if list
      (list-builder (.list b list) (:layers opts) opts)
      b)))

(defn neural-net-configuration 
  [opts]
  (.build (builder opts)))

(defn to-json [^NeuralNetConfiguration cfg]
  (.toJson cfg))

(defn to-edn [^NeuralNetConfiguration cfg]
  (json/read-str (.toJson cfg)
                 :key-fn #(keyword (camel-to-dashed %))))

(defn from-edn [cfg]
  (neural-net-configuration cfg))

(comment

  (opt/values)
  
  (def cfg (neural-net-configuration {:layer (dl4clj.nn.conf.layers.graves-lstm/graves-lstm-layer)
                                      :optimization-algo :stochastic-gradient-descent}))
  (print (to-json cfg))
  (neural-net-configuration (to-edn cfg))
  
  (json/read-str cfg-json :key-fn #(keyword (camel-to-dashed %)))


  (def opt {:optimization-algo :stochastic-gradient-descent
            :num-iterations 1
            :learning-rate 0.95
            :rms-decay 0.23
            :seed 1234
            :regularization true
            :l2 0.001
            ;; :layer (dl4clj.nn.conf.layers.graves-lstm/graves-lstm-layer)
            :layer (dl4clj.nn.conf.layers.rnn-output-layer/rnn-output-layer
                        {:loss-function :reconstruction-crossentropy
                         :activation "softmax"
                         :adam-mean-decay 0.3
                         :adam-var-decay 0.5
                         :bias-init 0.3
                         :dist (dl4clj.nn.conf.distribution.binomial-distribution/binomial-distribution 10 0.4)
                         :drop-out 0.01
                         :gradient-normalization :clip-l2-per-layer
                         :gradient-normalization-threshold 0.1
                         :l1 0.02
                         :l2 0.002
                         :learning-rate 0.95
                         :learning-rate-after {1000 0.5}
                         :learning-rate-score-based-decay-rate 0.001
                         :momentum 0.9
                         :momentum-after {10000 1.5}
                         :name "test"
                         :rho 0.5
                         :rms-decay 0.01
                         :updater :adam
                         :weight-init :distribution
                         :n-in 30
                         :n-out 30
                         :forget-gate-bias-init 0.12})
            ;; :list 3
            ;; :layers {0 (dl4clj.nn.conf.layers.graves-lstm/graves-lstm
            ;;             {:activation "softmax"
            ;;              :adam-mean-decay 0.3
            ;;              :adam-var-decay 0.5
            ;;              :bias-init 0.3
            ;;              :dist (dl4clj.nn.conf.distribution.binomial-distribution/binomial-distribution 10 0.4)
            ;;              :drop-out 0.01
            ;;              :gradient-normalization :clip-l2-per-layer
            ;;              :gradient-normalization-threshold 0.1
            ;;              :l1 0.02
            ;;              :l2 0.002
            ;;              :learning-rate 0.95
            ;;              :learning-rate-after {1000 0.5}
            ;;              :learning-rate-score-based-decay-rate 0.001
            ;;              :momentum 0.9
            ;;              :momentum-after {10000 1.5}
            ;;              :name "test"
            ;;              :rho 0.5
            ;;              :rms-decay 0.01
            ;;              :updater :adam
            ;;              :weight-init :distribution
            ;;              :n-in 30
            ;;              :n-out 30
            ;;              :forget-gate-bias-init 0.12})
            ;;          1 (dl4clj.nn.conf.layers.graves-lstm/graves-lstm
            ;;             {:activation "softmax"
            ;;              :adam-mean-decay 0.3
            ;;              :adam-var-decay 0.5
            ;;              :bias-init 0.3
            ;;              :dist (dl4clj.nn.conf.distribution.binomial-distribution/binomial-distribution 10 0.4)
            ;;              :drop-out 0.01
            ;;              :gradient-normalization :clip-l2-per-layer
            ;;              :gradient-normalization-threshold 0.1
            ;;              :l1 0.02
            ;;              :l2 0.002
            ;;              :learning-rate 0.95
            ;;              :learning-rate-after {1000 0.5}
            ;;              :learning-rate-score-based-decay-rate 0.001
            ;;              :momentum 0.9
            ;;              :momentum-after {10000 1.5}
            ;;              :name "test"
            ;;              :rho 0.5
            ;;              :rms-decay 0.01
            ;;              :updater :adam
            ;;              :weight-init :distribution
            ;;              :n-in 30
            ;;              :n-out 30
            ;;              :forget-gate-bias-init 0.12})
            ;;          2 (dl4clj.nn.conf.layers.rnn-output-layer/rnn-output-layer
            ;;             {:loss-function :reconstruction-crossentropy
            ;;              :activation "softmax"
            ;;              :adam-mean-decay 0.3
            ;;              :adam-var-decay 0.5
            ;;              :bias-init 0.3
            ;;              :dist (dl4clj.nn.conf.distribution.binomial-distribution/binomial-distribution 10 0.4)
            ;;              :drop-out 0.01
            ;;              :gradient-normalization :clip-l2-per-layer
            ;;              :gradient-normalization-threshold 0.1
            ;;              :l1 0.02
            ;;              :l2 0.002
            ;;              :learning-rate 0.95
            ;;              :learning-rate-after {1000 0.5}
            ;;              :learning-rate-score-based-decay-rate 0.001
            ;;              :momentum 0.9
            ;;              :momentum-after {10000 1.5}
            ;;              :name "test"
            ;;              :rho 0.5
            ;;              :rms-decay 0.01
            ;;              :updater :adam
            ;;              :weight-init :distribution
            ;;              :n-in 30
            ;;              :n-out 30
            ;;              :forget-gate-bias-init 0.12})}
            })

  (def cfg (neural-net-configuration opt))

  (require '[dl4clj.examples.example-utils :refer (+default-character-set+)])
  (require '[dl4clj.nn.conf.layers.graves-lstm :refer (graves-lstm)])
  (require '[dl4clj.nn.conf.layers.rnn-output-layer :refer (rnn-output-layer)])
  (require '[dl4clj.nn.conf.distribution.uniform-distribution :refer (uniform-distribution)])

  ;; works: 
  (let [opt {:optimization-algo :stochastic-gradient-descent
             :num-iterations 1
             :learning-rate 0.1
             :rmsDecay 0.95
             :seed 12345
             :regularization true
             :l2 0.001
             ;; :list 3
             :layer (graves-lstm
                     {:nIn (count +default-character-set+)
                      :nOut 200
                      :updater :rmsprop
                      :activation "tanh"
                      :weight-init :distribution
                      :dist (uniform-distribution -0.01 0.01)})
             :pretrain false
             :backprop true}]
    (def cfg (neural-net-configuration opt)))
  
  ;; doesn't work:
  (let [opt {:optimization-algo :stochastic-gradient-descent
             :num-iterations 1
             :learning-rate 0.1
             :rmsDecay 0.95
             :seed 12345
             :regularization true
             :l2 0.001
             :list 1
             :layers {0 (graves-lstm
                         {:nIn (count +default-character-set+)
                          :nOut 200
                          :updater :rmsprop
                          :activation "tanh"
                          :weight-init :distribution
                          :dist (uniform-distribution -0.01 0.01)})
                      ;; 1 (graves-lstm
                      ;;    {:nIn 200
                      ;;     :nOut 200
                      ;;     :updater :rmsprop
                      ;;     :activation "tanh"
                      ;;     :weight-init :distribution
                      ;;     :dist (uniform-distribution -0.01 0.01)})
                      ;; 2 (rnn-output-layer
                      ;;    {:loss-function :mcxent
                      ;;     :updater :rmsprop
                      ;;     :n-in 200
                      ;;     :n-out (count +default-character-set+)
                      ;;     :weight-init :distribution
                      ;;     :dist (uniform-distribution -0.01 0.01)})
                      }
             :pretrain false
             :backprop true}]
    (def cfg (neural-net-configuration opt)))


  

  

  
  (to-edn cfg)
  (print (to-json cfg) )

  ;;; ???
  ;;; - edn doesn't show the layers
  ;;; - is layer needed (check in example)

  )
  

;; NeuralNetConfiguration.ListBuilder	list(int size)
;; Number of layers not including input.

