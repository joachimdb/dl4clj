(ns ^{:doc ""}
  dl4clj.examples.word2vec.word2vec-raw-text-example
  (:import [org.deeplearning4j.models.word2vec Word2Vec Word2Vec$Builder]
           [org.deeplearning4j.models.word2vec.wordstore.inmemory InMemoryLookupCache]
           [org.deeplearning4j.models.embeddings.loader WordVectorSerializer]
           [org.deeplearning4j.text.sentenceiterator BasicLineIterator]
           [org.deeplearning4j.text.sentenceiterator SentenceIterator]
           [org.deeplearning4j.text.sentenceiterator UimaSentenceIterator]
           [org.deeplearning4j.text.tokenization.tokenizer.preprocessor CommonPreprocessor]
           [org.deeplearning4j.text.tokenization.tokenizerfactory DefaultTokenizerFactory]
           [org.deeplearning4j.text.tokenization.tokenizerfactory TokenizerFactory]
           [org.deeplearning4j.ui UiServer]))


(defn ^Word2Vec vec-from-file [^String path {:keys [min-word-frequency
                                                    iterations
                                                    layer-size
                                                    seed
                                                    window-size]
                                             :or {min-word-frequency 5
                                                  iterations 1
                                                  layer-size 100
                                                  seed 42
                                                  window-size 5}}]
  (let [iter (BasicLineIterator. path) ;; Strip white space before and after for each line
        tf (doto (DefaultTokenizerFactory.) ;; Split on white spaces in the line to get words
             (.setTokenPreProcessor (CommonPreprocessor.)))]
    (-> (Word2Vec$Builder.)
        (.iterations iterations)
        (.minWordFrequency min-word-frequency)
        (.layerSize layer-size)
        (.seed seed)
        (.windowSize window-size)
        (.iterate iter)
        (.tokenizerFactory tf)
        (.build)
        (.fit))))

(defn save [^Word2Vec model path]
  (WordVectorSerializer/writeWordVectors model path))

(defn n-closest [n ^Word2Vec model ^String target n]
  (seq (.wordsNearest model target (int n))))

(defn ui-server []
  (UiServer/getInstance))

(comment 

  ;; Example usage:
  (def vec (vec-from-file "/tmp/

  (.getPort ui-server)

)
