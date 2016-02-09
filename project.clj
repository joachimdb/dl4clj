(defproject dl4clj "0.1.0-SNAPSHOT"
  :description "DL4J's Iris example straight port to Clojure"
  :dependencies [[org.clojure/clojure "1.7.0"]
                 [org.deeplearning4j/deeplearning4j-core "0.4-rc3.4"]
                 [org.apache.commons/commons-io "1.3.2"]
                 [org.nd4j/nd4j-jblas "0.4-rc3.5"]]             ;;; MacBook requirement
  :main ^:skip-aot dl4j-clj-example.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
