(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/distribution/NormalDistribution.html"}
  dl4clj.nn.conf.distribution.normal-distribution
  (:import [org.deeplearning4j.nn.conf.distribution NormalDistribution]))


(defn normal-distribution [mean std]
  (NormalDistribution. mean std))

(comment
  
  (normal-distribution 10 0.3)
  
)
