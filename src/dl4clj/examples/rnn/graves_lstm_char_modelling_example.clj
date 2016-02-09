(ns ^{:doc "

GravesLSTM Character modelling example

@author Joachim De Beule, based on Alex Black's java code, see https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/rnn/GravesLSTMCharModellingExample.java

For general instructions using deeplearning4j's implementation of recurrent neural nets see http://deeplearning4j.org/usingrnns.html
"}
  dl4clj.examples.rnn.graves-lstm-char-modelling-example
  (:require [dl4clj.examples.rnn.character-iterator :refer :all])
  (:import [java.io File IOException]
           [java.net URL]
           [java.text CharacterIterator]
           [java.nio.charset Charset]
           [java.util Random]
           
           [org.deeplearning4j.datasets.iterator DataSetIterator]

           [org.deeplearning4j.nn.api Layer OptimizationAlgorithm]
           [org.deeplearning4j.nn.conf MultiLayerConfiguration Updater NeuralNetConfiguration NeuralNetConfiguration$Builder]
           [org.deeplearning4j.nn.conf.distribution UniformDistribution]
           [org.deeplearning4j.nn.conf.layers GravesLSTM GravesLSTM$Builder RnnOutputLayer RnnOutputLayer$Builder]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.nd4j.linalg.api.ndarray INDArray]
           [org.nd4j.linalg.factory Nd4j]
           [org.nd4j.linalg.lossfunctions LossFunctions LossFunctions$LossFunction]))

(defn- output-distribution 
  "Utility fn to convert a 1 dimensional NDArray to an array of doubles."
  [^INDArray output]
  (let [d (double-array (.length output))]
    (dotimes [i (.length output)]
      (aset d i (.getDouble output i)))
    d))

(defn- sample-from-distribution 
  "Given a probability distribution over discrete classes (an array of doubles), sample from the
  distribution and return the generated class index.

  @param distribution Probability distribution over classes. Must sum to 1.0.
"
  [^"[D" distribution] 
  (let [toss (rand)]
    (loop [i 0
           sum (aget distribution 0)]
      (cond (<= toss sum) i
            (< i (count distribution)) (recur (inc i) (+ sum (aget distribution i)))
            :else (throw (IllegalArgumentException. (str "Distribution is invalid? toss= " toss ", sum=" sum)))))))

(defn generate 
  "Generate a number of samples sample given an initialization string for 'priming' the RNN with a
  sequence to extend/continue."
  [^MultiLayerNetwork net initialization-string {:keys [valid-characters 
                                                        chars-per-sample 
                                                        num-samples]
                                                 :or {valid-characters +default-character-set+
                                                      num-samples 4
                                                      chars-per-sample 10}
                                                 :as opts}]
  (assert (not (empty? initialization-string)) "initialization string cannot be empty")
  (let [char-to-idx (index-map valid-characters)
        idx-to-char (zipmap (vals char-to-idx) (keys char-to-idx))
        initialization-input (Nd4j/zeros (int-array [num-samples (count valid-characters) (count initialization-string)]))
        sb (for [i (range num-samples)] (StringBuilder. ^String initialization-string))]
    
    ;; Fill input for initialization
    (dotimes [i (count initialization-string)]
      (let [idx (char-to-idx (nth initialization-string i))]
        (dotimes [j num-samples]
          (.putScalar ^INDArray initialization-input (int-array [j idx i]) (float 1.0)))))
    
    (.rnnClearPreviousState net)
    (loop [i 0 
           output (.tensorAlongDimension (.rnnTimeStep net initialization-input)
                                         (int (dec (count initialization-string)))
                                         (int-array [1 0]))]
      ;; Set up next input (single time step) by sampling from previous output
      (let [next-input (Nd4j/zeros (int num-samples) (int (count valid-characters)))]
        (dotimes [s num-samples]
          (let [sampled-character-idx (sample-from-distribution (output-distribution (.slice output s 1)))]
            (.putScalar next-input (int-array [s sampled-character-idx]) (float 1.0))
            (.append ^StringBuilder (nth sb s) (idx-to-char sampled-character-idx)))) ;; Add sampled character to StringBuilder (human readable output)
        (when (< i chars-per-sample)
          (recur (inc i)
                 (.rnnTimeStep net next-input)))))
    
    (map #(.toString ^StringBuilder %) sb)))

(defn graves-lstm-char-modelling-net 
  "Builds and returns an LSTM net."
  [{:keys [valid-characters
           lstm-layer-size ;; Number of units in each GravesLSTM layer
           learning-rate
           rms-decay
           iterations
           seed]
    :or {valid-characters +default-character-set+
         lstm-layer-size 200
         learning-rate 0.1
         rms-decay 0.95
         iterations 1
         seed 12345}
    :as opts}]
  (let [conf (-> (NeuralNetConfiguration$Builder.)
                 (.optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
                 (.iterations (int iterations))
                 (.learningRate (double learning-rate))
                 (.rmsDecay (double rms-decay))
                 (.seed (int seed))
                 (.regularization true)
                 (.l2 0.001)
                 (.list (int 3))
                 (.layer 0 (-> (GravesLSTM$Builder.)
                               (.nIn (count valid-characters))
                               (.nOut (int lstm-layer-size))
                               (.updater Updater/RMSPROP)
                               (.activation "tanh")
                               (.weightInit WeightInit/DISTRIBUTION)
                               (.dist (UniformDistribution. -0.08 0.08))
                               (.build)))
                 (.layer 1  (-> (GravesLSTM$Builder.)
                                (.nIn (int lstm-layer-size))
                                (.nOut (int lstm-layer-size))
                                (.updater Updater/RMSPROP)
                                (.activation "tanh")
                                (.weightInit WeightInit/DISTRIBUTION)
                                (.dist (UniformDistribution. -0.08 0.08))
                                (.build)))
                 (.layer 2  (-> (RnnOutputLayer$Builder. (LossFunctions$LossFunction/MCXENT))
                                (.nIn (int lstm-layer-size))
                                (.nOut (count valid-characters))
                                (.activation "softmax")
                                (.updater Updater/RMSPROP)
                                (.weightInit WeightInit/DISTRIBUTION)
                                (.dist (UniformDistribution. -0.08 0.08))
                                (.build)))
                 (.pretrain false)
                 (.backprop true)
                 (.build))
        net (MultiLayerNetwork. conf)]
    (.init net)
    (.setListeners net [(ScoreIterationListener. (int 1))])
    
    ;; Print the  number of parameters in the network (and for each layer)
    (dotimes [i (count (.getLayers net))]
      (println "Number of parameters in layer "  i  ": "  (.numParams (nth (.getLayers net) i))))
    (println "Total number of network parameters: " (reduce + (map #(.numParams %) (.getLayers net))))
    
    net))

(defn train 
  "Performs a number of training epochs of an LSTM net on examples from a character-iterator. Prints
  generated samples in between epochs."
    [^MultiLayerNetwork net ^DataSetIterator character-dataset-iterator {:keys [epochs 
                                                                                generation-initialization-string 
                                                                                num-samples
                                                                                chars-per-sample
                                                                                valid-characters]
                                                                         :or {epochs 1
                                                                              num-samples 5
                                                                              chars-per-sample 50
                                                                              valid-characters +default-character-set+}
                                                                         :as opts}]
  (dotimes [i epochs]
    (.reset character-dataset-iterator)
    (.fit net character-dataset-iterator)
    (println "--------------------");
    (println "Completed epoch " i );
    (println (str "Sampling characters from network given initialization \"" generation-initialization-string "\""))
    (doseq [sample (generate net (or generation-initialization-string
                                     (str (rand-nth (seq valid-characters))))
                             opts)]
      (println "Sample: " sample)
      (println)))
  net)





(comment 

  ;;; Example usage:

  (def iter (shakespeare-iterator {:valid-characters +default-character-set+ 
                                   :batch-size 32
                                   :chars-per-segment 100
                                   :max-segments (* 32 50)}))

  (def net (graves-lstm-char-modelling-net {:valid-characters +default-character-set+ 
                                            :lstm-layer-size 200
                                            :iterations 1
                                            :learning-rate 0.1
                                            :rms-decay 0.95
                                            :seed 12345}))

  (train net iter {:valid-characters +default-character-set+
                   :epochs 30
                   :num-samples 4
                   :chars-per-sample 300})

  )

