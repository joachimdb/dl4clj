(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/distribution/BinomialDistribution.html"}
  dl4clj.nn.conf.distribution.binomial-distribution
  (:import [org.deeplearning4j.nn.conf.distribution BinomialDistribution]))


(defn binomial-distribution [number-of-trials probability-of-success]
  (BinomialDistribution. number-of-trials probability-of-success))

(comment
  
  (binomial-distribution 10 0.3)
  
)
