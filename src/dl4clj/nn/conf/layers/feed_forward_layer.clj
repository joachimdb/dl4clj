(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/FeedForwardLayer.html"}
  dl4clj.nn.conf.layers.feed-forward-layer
  (:require [dl4clj.nn.conf.layers.layer :as layer])
  (:import [org.deeplearning4j.nn.conf.layers FeedForwardLayer$Builder]))

(defn builder [^FeedForwardLayer$Builder builder {:keys [n-in 
                                                         n-out]
                                                  :or {}
                                                  :as opts}]
  (layer/builder builder opts)
  (when n-in
    (.nIn builder n-in))
  (when n-out
    (.nOut builder n-out))
  builder)


