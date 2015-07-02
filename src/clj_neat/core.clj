(ns clj-neat.core
  (:require [clojure.core.async
             :as a
             :refer [>! <! >!! <!! go chan buffer close! thread
                     alts! alts!! timeout]])
  (:import [java.lang Math]))

(defrecord Gene [in out weight enabled innovation])
(defrecord Neuron [incoming value])
(defrecord Network [inputs hidden outputs])
(defrecord MutationRates [connections link bias node enable disable step])
(defrecord Genome [genes fitness adjusted-fitness network max-neuron global-rank mutation-rates])
(defrecord Species [top-fitness staleness genomes average-fitness])
(defrecord Pool [species generation innovation current-species current-genome current-frame max-fitness])

(declare inputs outputs perturb-chance max-nodes)

(defn sigmoid [x]
  (- (/ 2 (inc (Math/exp (* -4.9 x)))) 1))

(defn new-pool []
  (map->Pool {:species []
              :generation 0
              :innovation 0
              :current-species 1
              :current-genome 1
              :current-frame 0
              :max-fitness 0}))

(def pool (atom (new-pool)))

(defn new-species []
  (map->Species {:top-fitness 0
                 :staleness 0
                 :genomes []
                 :average-fitness 0}))

(defn new-genome []
  (map->Genome {:genes []
                :fitness 0
                :adjusted-fitness 0
                :network []
                :max-neuron 0
                :global-rank 0
                :mutation-rates (map->MutationRates {:connections 0.25
                                                     :link 2.00
                                                     :bias 0.40
                                                     :node 0.50
                                                     :enable 0.20
                                                     :disable 0.40
                                                     :step 0.10})}))

(defn new-gene []
  (map->Gene {:in 0
              :out 0
              :weight 0.0
              :enabled true
              :innovation 0}))

(defn new-neuron []
  (map->Neuron {:incoming []
                :value 0.0}))

(defn new-innovation! []
  (swap! pool #(assoc %1 :innovation (inc (:innovation %1))))
  (:innovation @pool))

(defn new-network []
  (map->Network
   {:inputs (into (sorted-map) (zipmap (range inputs) (repeatedly inputs new-neuron)))
    :hidden (sorted-map)
    :outputs (into (sorted-map) (zipmap (range inputs (+ inputs outputs)) (repeatedly outputs new-neuron)))}))

(defn get-neuron [^Network network id]
  (get (reduce merge (vals network)) id))

(defn gen-network [^Genome genome]
  (let [network (new-network)
        genes (sort-by :out (:genes genome))
        gene-assigner (fn [^Network network ^Gene gene]
                        (let [neuron (get-neuron network (:out gene))
                              neuron (if neuron neuron (new-neuron))
                              network (if (get-neuron network (:in gene)) network (assoc-in network [:hidden (:in gene)] (new-neuron)))]
                          (assoc-in network [(if (< (:out gene) inputs)
                                               :inputs
                                               (if (< (:out gene) (+ inputs outputs))
                                                 :outputs
                                                 :hidden)) (:out gene)] (assoc neuron :incoming (conj (:incoming neuron) gene)))))]
    (assoc genome :network (if (empty? genes) network (reduce gene-assigner network genes)) :genes genes)))

(defn eval-network [^Network network input-vals]
  (if (not= (count input-vals) inputs)
    []
    (let [network (assoc network :inputs (zipmap (range inputs) (map #(assoc %1 :value %2) (vals (:inputs network)) input-vals)))
          neuron-changer (fn [^Network my-network [neuron-key ^Neuron neuron]]
                           (let [sum (reduce #(+ %1 (* (:weight %2) (:value (get-neuron my-network (:in %2))))) 0 (:incoming neuron))]
                             (if (> (count (:incoming neuron)) 0)
                               (assoc-in my-network [(if (< neuron-key (+ inputs outputs))
                                                       :outputs
                                                       :hidden) neuron-key] (assoc neuron :value (sigmoid sum)))
                               my-network)))
          network (reduce neuron-changer network (:hidden network))
          network (reduce neuron-changer network (:outputs network))
          output-neurons (vals (sort-by key < (:outputs network)))]
      (mapv #(> (:value %1) 0) output-neurons))))

(defn crossover [^Genome genome1 ^Genome genome2]
  (let [fittest? (> (:fitness genome2) (:fitness genome1))
        g1 (if fittest? genome2 genome1)
        g2 (if fittest? genome1 genome2)
        innovations (group-by :innovation (:genes g2))
        new-gene (fn [gene1]
                   (let [gene2 (first (get innovations (:innovation gene1)))]
                     (if (and gene2 (= (rand-int 1) 1) (:enabled gene2))
                       gene2
                       gene1)))
        genes (map new-gene (:genes g1))]
    (assoc (new-genome) :genes genes :max-neuron (max (:max-neuron g1) (:max-neuron g2)) :mutation-rates (:mutation-rates g1))))

(defn random-neuron [network non-input?]
  (let [neurons (reduce merge (vals network))]
    (if non-input?
      (+ inputs (rand-int (- (count neurons) inputs)))
      (rand-int (count neurons)))))

(defn contains-link? [genes link]
  (not (empty? (filter #(and (= (:in %1) (:in link)) (= (:out %1) (:out link))) genes))))

(defn point-mutate [^Genome genome]
  (let [step (:step (:mutation-rates genome))
        gene-mutate (fn [^Gene gene]
                      (if (< (rand) perturb-chance)
                        (assoc gene :weight (- (+ (:weight gene) (* (rand) step 2)) step))
                        (assoc gene :weight (- (* (rand) 4) 2))))]
    (assoc genome :genes (map gene-mutate (:genes genome)))))

(defn link-mutate [^Genome genome force-bias?]
  (let [base-neuron1 (random-neuron (:network genome) false)
        base-neuron2 (random-neuron (:network genome) true)
        condition? (and (>= base-neuron1 inputs) (< base-neuron1 (+ inputs outputs)))
        neuron1 (if condition? base-neuron2 base-neuron1)
        neuron2 (if condition? base-neuron1 base-neuron2)
        link (assoc (new-gene) :in (if force-bias? (dec inputs) neuron1) :out neuron2)]
    (if (and (not= (:in link) (:out link)) (not (and (>= neuron1 inputs) (< neuron1 (+ inputs outputs)))) (not (contains-link? (:genes genome) link)))
      (let [link (assoc link :innovation (new-innovation!) :weight (- (* 4 (rand)) 2))]
        (gen-network (assoc genome :genes (conj (:genes genome) link))))
      (recur genome force-bias?))))

(defn node-mutate [^Genome genome]
  (if (empty? (:genes genome))
    genome
    (let [rand-gene (rand-int (count (:genes genome)))
          gene (nth (:genes genome) rand-gene)]
      (if (:enabled gene)
        (let [gene1 (assoc gene :out (+ inputs outputs (:max-neuron genome)) :weight 1.0 :innovation (new-innovation!) :enabled true)
              gene2 (assoc gene :in (+ inputs outputs (:max-neuron genome)) :innovation (new-innovation!) :enabled true)
              genome (assoc genome :genes (into [] (concat (into [] (take rand-gene (:genes genome))) [(assoc gene :enabled false)] (into [] (drop (inc rand-gene) (:genes genome))))))]
          (gen-network (assoc genome :genes (conj (:genes genome) gene1 gene2) :max-neuron (inc (:max-neuron genome)))))
        (recur genome)))))

(defn enable-disable-mutate [^Genome genome enable?]
  (let [candidates (filter #(not= (:enabled %1) enable?) (:genes genome))]
    (if (empty? candidates)
      genome
      (let [candidate (rand-int (count (:genes genome)))]
        (if (not= (:enabled (nth (:genes genome) candidate)) enable?)
          (assoc genome :genes (into [] (concat (into [] (take candidate (:genes genome))) [(assoc (nth (:genes genome) candidate) :enabled enable?)] (into [] (drop (inc candidate) (:genes genome))))))
          (recur genome enable?))))))

(defn do-rate [rate func & args]
  (if (<= rate 0.0)
    (first args)
    (if (< (rand) rate)
      (let [result (apply func args)]
        (recur (dec rate) func (flatten [result (rest args)])))
      (recur (dec rate) func args))))

(defn mutate [^Genome genome]
  (let [change-rate (fn [rate]
                      (if (= (rand-int 2) 0)
                        (* rate 0.95)
                        (* rate 1.05263)))
        rates (map->MutationRates (reduce merge (map #(hash-map (key %1) (change-rate (val %1))) (:mutation-rates genome))))
        genome (assoc genome :mutation-rates rates)
        do-rate-genome (fn [genome rate func & args]
                         (apply do-rate (get-in genome [:mutation-rates rate]) func genome args))]
    (-> genome
        (#(if (< (rand) (get-in %1 [:mutation-rates :connections]))
            (point-mutate %1)
            %1))
        (do-rate-genome :link link-mutate false)
        (do-rate-genome :bias link-mutate true)
        (do-rate-genome :node node-mutate)
        (do-rate-genome :enable enable-disable-mutate true)
        (do-rate-genome :disable enable-disable-mutate false))))

(defn basic-genome []
  (-> (new-genome)
      (gen-network)
      (mutate)))

(def inputs 3)
(def outputs 3)

(def perturb-chance 0.90)

(def max-nodes 1000000)

(def sample-genome (basic-genome))
