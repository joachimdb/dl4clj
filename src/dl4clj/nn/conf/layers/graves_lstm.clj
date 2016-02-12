(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/GravesLSTM.html"}
  dl4clj.nn.conf.layers.graves-lstm
  (:require [dl4clj.nn.conf.layers.base-recurrent-layer :as br-layer])
  (:import [org.deeplearning4j.nn.conf.layers GravesLSTM$Builder]))

(defn builder [{:keys [forget-gate-bias-init] ;; (double)
                :or {}
                :as opts}]
  (let [builder ^GravesLSTM$Builder (br-layer/builder (GravesLSTM$Builder.) opts)]
    (when forget-gate-bias-init
      (.forgetGateBiasInit builder forget-gate-bias-init))
    builder))

(defn graves-lstm
  ([]
   (graves-lstm {}))
  ([{:keys [forget-gate-bias-init] ;; (double)
    :or {}
    :as opts}]
  (.build ^GravesLSTM$Builder (builder opts))))


(comment

  ;; Example usages:

  (graves-lstm-layer)
  
  (graves-lstm-layer {:activation "softmax"
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
                      :weight-init :normalized
                      :n-in 30
                      :n-out 30
                      :forget-gate-bias-init 0.12})

  )
