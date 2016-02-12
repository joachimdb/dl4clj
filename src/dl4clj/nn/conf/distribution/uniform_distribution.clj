(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/distribution/UniformDistribution.html"}
  dl4clj.nn.conf.distribution.uniform-distribution
  (:import [org.deeplearning4j.nn.conf.distribution UniformDistribution]))


(defn uniform-distribution [lower upper]
  (UniformDistribution. lower upper))

(comment
  
  (uniform-distribution 0.3 10)
  
)
