(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/RNNOutputLayer.html"}
  dl4clj.nn.conf.layers.rnn-output-layer
  (:require [dl4clj.nn.conf.layers.base-output-layer :as base-out-layer]
            [nd4clj.linalg.lossfunctions.loss-functions :as loss-functions])
  (:import [org.deeplearning4j.nn.conf.layers RnnOutputLayer RnnOutputLayer$Builder]
           [org.nd4j.linalg.lossfunctions LossFunctions$LossFunction]))

(defn builder [^RnnOutputLayer$Builder b opts]
  (base-out-layer/builder b opts))

(defn rnn-output-layer [{:keys [loss-function]
                         :or {}
                         :as opts}]
  (.build ^RnnOutputLayer$Builder (builder (RnnOutputLayer$Builder. ^LossFunctions$LossFunction (loss-functions/value-of loss-function)) opts)))


(comment
  
  (rnn-output-layer {:loss-function :reconstruction-crossentropy})

  )


