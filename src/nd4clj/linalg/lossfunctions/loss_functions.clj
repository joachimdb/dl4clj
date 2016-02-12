(ns ^{:doc "see http://nd4j.org/doc/org/nd4j/linalg/lossfunctions/LossFunctions.html"}
  nd4clj.linalg.lossfunctions.loss-functions
  (:import [org.nd4j.linalg.lossfunctions LossFunctions LossFunctions$LossFunction ]))


(defn value-of [k]
  (if (string? k)
    (LossFunctions$LossFunction/valueOf k)
    (LossFunctions$LossFunction/valueOf (clojure.string/replace (clojure.string/upper-case (name k)) "-" "_"))))

(defn values []
  (map #(keyword (clojure.string/replace (clojure.string/lower-case (str %)) "_" "-")) (LossFunctions$LossFunction/values)))

(comment

  (values)
  (type (value-of :reconstruction-crossentropy))
  )
