#!/usr/bin/env ruby
# frozen_string_literal: true

# =============================================================================
# Individual-based Hawk–Dove game simulator
# =============================================================================
#
# Usage:
#   ruby hawk_dove_sim.rb                    
#   ruby hawk_dove_sim.rb --generations 1000 
#   ruby hawk_dove_sim.rb --no-csv           
# =============================================================================

require 'csv'
require 'fileutils'


# MathUtil: Gaussian random numbers using the Box–Muller method
module MathUtil
  def self.rand_gaussian(mean, sigma)
    u1 = [rand, Float::EPSILON].max  # Avoid log(0)
    u2 = rand
    z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
    mean + sigma * z
  end
end


Config = Struct.new(
  :n,                  # Population size
  :v,                  # Resource value
  :c,                  # Cost of fighting
  :mu,                 # Mutation rate
  :mut_sigma,          # Mutation scale (standard deviation of ±Δ)
  :max_battles,        # Maximum number of interactions per generation
  :k_threshold,        # Early termination threshold for k (should be set to < N/2)
  :generations,        # Total number of generations
  :log_interval,       # Console logging interval (in generations)
  :csv_path,           # CSV output path (nil = disabled)
  :viz_dir,            # PNG output directory (nil = disabled)
  :viz_snapshots,      # Number of gene_p snapshot histograms to render
  keyword_init: true
) do
  # Default configuration
  def self.default
    new(
      n:                100,
      v:                0.4,
      c:                0.6,
      mu:               0.01,
      mut_sigma:        0.05,
      max_battles:      500,
      k_threshold:      40,
      generations:      200,
      log_interval:     10,
      csv_path:         "results.csv",
      viz_dir:          ".",
      viz_snapshots:    6
    )
  end


  def ess
    v / c
  end
end


class Organism
  attr_accessor :gene_p, :fitness

  # gene_p: probability of playing Hawk [0.0, 1.0]
  def initialize(gene_p)
    @gene_p  = gene_p.clamp(0.0, 1.0)
    @fitness = 0.0
  end
 
  def play
    rand < @gene_p ? :hawk : :dove
  end

  def fitness_floor
    @fitness.floor
  end

  # Reproduce offspring: generate floor(fitness) offspring, with mutation occurring with probability mu
  # Assumes fitness >= 1.0 before calling this method
  def reproduce(mu, mut_sigma)
    Array.new(fitness_floor) do
      new_p = if rand < mu
                MathUtil.rand_gaussian(@gene_p, mut_sigma).clamp(0.0, 1.0)
              else
                @gene_p
              end
      Organism.new(new_p)
    end
  end
end


module HawkDoveGame
  # Let two individuals interact and add the resulting payoffs to their fitness
  def self.resolve(org_a, org_b, v, c)
    sa = org_a.play
    sb = org_b.play
    pa, pb = payoffs(sa, sb, v, c)
    org_a.fitness += pa
    org_b.fitness += pb
  end

  # k = sum of floor(fitness) for all individuals with fitness >= 1
  # k determines the number of individuals replaced in the next generation (= selection pool size)
  def self.compute_k(individuals)
    individuals.sum { |ind| ind.fitness >= 1.0 ? ind.fitness_floor : 0 }
  end

  private_class_method def self.payoffs(sa, sb, v, c)
    case [sa, sb]
    in [:hawk, :hawk] then [(v - c) / 2.0, (v - c) / 2.0]
    in [:hawk, :dove] then [v,              0.0           ]
    in [:dove, :hawk] then [0.0,            v             ]
    in [:dove, :dove] then [v / 2.0,        v / 2.0       ]
    end
  end
end


class Population
  attr_reader :individuals, :last_k, :last_early_stop

  def initialize(config)
    @config = config
    # Initial population: gene_p is initialized from a uniform random distribution over [0, 1]
    @individuals   = Array.new(config.n) { Organism.new(rand) }
    @last_k        = 0
    @last_early_stop = false
  end

  # Advance one generation and update @individuals to the next generation
  def step
    reset_fitness
    @last_early_stop = run_battles
    @last_k          = evolve
  end

  # Return summary statistics (for display and CSV output)
  def stats
    ps        = @individuals.map(&:gene_p)
    n         = ps.size
    mean_p    = ps.sum / n
    variance  = ps.sum { |p| (p - mean_p)**2 } / n
    sd_p      = Math.sqrt(variance)
    {
      mean_p:    mean_p,
      sd_p:      sd_p,
      hawk_freq: mean_p,  # Expected value of gene_p = frequency of Hawk play in the population
      n:         n
    }
  end

  private

  def reset_fitness
    @individuals.each { |ind| ind.fitness = 0.0 }
  end

  
  # Interaction phase: repeatedly perform interactions with random pairings (with replacement)
  # Track approx_k using incremental updates (O(1) per interaction) for triggering,
  # and terminate early when it reaches k_threshold
  def run_battles
    cfg        = @config
    approx_k   = 0
    early_stop = false

    cfg.max_battles.times do
      # Sample two individuals with replacement; skip if identical (avoid self-interaction)
      a = @individuals.sample
      b = @individuals.sample
      next if a.equal?(b)

      k_before = floor_k_of(a) + floor_k_of(b)

      HawkDoveGame.resolve(a, b, cfg.v, cfg.c)

      approx_k += floor_k_of(a) + floor_k_of(b) - k_before

      if approx_k >= cfg.k_threshold
        early_stop = true
        break
      end
    end

    early_stop
  end

  # Return floor(fitness) if fitness >= 1, otherwise 0
  # Used for incremental approx_k updates
  def floor_k_of(org)
    org.fitness >= 1.0 ? org.fitness_floor : 0
  end

  # Selection and reproduction phase: remove the lowest k individuals and let survivors reproduce
  # Returns k (number of individuals replaced in this generation)
  def evolve
    cfg     = @config
    sorted  = @individuals.sort_by(&:fitness).reverse
    k       = HawkDoveGame.compute_k(sorted)

    survivors = sorted.first(cfg.n - k)

    # Clamp negative fitness → add 1 to all → generate floor(fitness) offspring
    # Σfloor(fi + 1) = Σfloor(fi) + (N-k) = k + (N-k) = N
    survivors.each { |ind| ind.fitness = [ind.fitness, 0.0].max + 1.0 }
    @individuals = survivors.flat_map { |ind| ind.reproduce(cfg.mu, cfg.mut_sigma) }

    k
  end
end


class Reporter
  def initialize(config)
    @config = config
    @csv    = config.csv_path ? CSV.open(config.csv_path, "w") : nil
    @csv&.<< %w[generation mean_p sd_p hawk_freq n k early_stop ess]
  end

  def print_header
    cfg = @config
    puts "=" * 65
    puts "  Hawk-Dove Evolution Simulator"
    puts "=" * 65
    puts "  N=#{cfg.n}  |  V=#{cfg.v}  C=#{cfg.c}  " \
         "|  mu=#{cfg.mu}  mut_sigma=#{cfg.mut_sigma}"
    puts "  max_battles=#{cfg.max_battles}  " \
         "k_threshold=#{cfg.k_threshold}  (incremental monitor)"
    puts "  ESS (theoretical p*) = #{format('%.4f', cfg.ess)}"
    puts "=" * 65
    puts format("  %-6s  %-8s  %-8s  %-6s  %-5s  %s",
                "Gen", "mean_p", "sd_p", "k", "n", "")
    puts "-" * 65
  end

  def report(gen, stats, k, early_stop)
    @csv&.<< [
      gen,
      stats[:mean_p].round(5), stats[:sd_p].round(5),
      stats[:hawk_freq].round(5), stats[:n],
      k, early_stop, @config.ess.round(5)
    ]

    return unless gen % @config.log_interval == 0 || gen == 1

    flag = early_stop ? " *" : "  "
    puts format("  %6d  %8.4f  %8.4f  %6d  %5d  %s",
                gen, stats[:mean_p], stats[:sd_p], k, stats[:n], flag)
  end

  def print_footer(final_stats)
    cfg = @config
    puts "-" * 65
    puts "  Final  mean_p=#{format('%.4f', final_stats[:mean_p])}  " \
         "ESS=#{format('%.4f', cfg.ess)}  " \
         "diff=#{format('%+.4f', final_stats[:mean_p] - cfg.ess)}"
    puts "=" * 65
    if @csv
      @csv.close
      puts "  CSV saved: #{cfg.csv_path}"
    end
    puts "  * = early stop (k reached threshold)"
  end
end

class Visualizer
  N_BINS    = 20
  BIN_WIDTH = 1.0 / N_BINS
 
 
  def initialize(config)
    @config         = config
    @k_history      = []
    @gene_p_history = []  # Array<Array<Float>>: gene_p values for all individuals across all generations
  end
 
  # Called every generation to accumulate data
  def record(_gen, gene_ps, k)
    @k_history      << k
    @gene_p_history << gene_ps.dup
  end
 
  def render
    require 'gruff'
    require 'rmagick'
 
    dir = @config.viz_dir
    FileUtils.mkdir_p(dir)
 
    render_k_history(dir)
    render_gene_p_final(dir)
    render_gene_p_snapshots(dir)
 
    puts "  Plots saved to: #{File.expand_path(dir)}/"
  end
 
  private
 
  def bin_index(p)
    [(p / BIN_WIDTH).floor, N_BINS - 1].min
  end
 
 # For gruff: map index to generation labels (sparse hash)
  def x_labels(size)
    step = [size / 10, 1].max
    (0...size).each_with_object({}) do |i, h|
      h[i] = (i + 1).to_s if i == 0 || (i + 1) % step == 0
    end
  end
 
  # Plot 1: k over generations (line plot)
  def render_k_history(dir)
    g = Gruff::Line.new(1000)
    g.title      = "Selection Intensity k per Generation"
    g.dot_radius = 0
    g.line_width = 1.5
    g.labels     = x_labels(@k_history.size)
    g.data("k", @k_history)
    g.data("threshold (#{@config.k_threshold})",
           Array.new(@k_history.size, @config.k_threshold))
    g.minimum_value = 0
    g.write(File.join(dir, "k_history.png"))
  end
 
  # Plot 2: gene_p histogram of the final generation (bar chart)
  def render_gene_p_final(dir)
    bins = Array.new(N_BINS, 0)
    @gene_p_history.last.each { |p| bins[bin_index(p)] += 1 }
 
    labels = N_BINS.times.each_with_object({}) do |i, h|
      h[i] = format("%.2f", i * BIN_WIDTH) if i % 4 == 0
    end
 
    g = Gruff::Bar.new(1000)
    g.title         = "Final Generation: gene_p Distribution" \
                      "  (ESS = #{format('%.3f', @config.ess)})"
    g.labels        = labels
    g.minimum_value = 0
    g.data("individuals", bins)
    g.write(File.join(dir, "gene_p_final.png"))
  end
 
  # Plot 3: gene_p snapshot histograms at evenly spaced generations (side by side)
  def render_gene_p_snapshots(dir)
    n_gen       = @gene_p_history.size
    n_snap      = @config.viz_snapshots.clamp(1, n_gen)
    snap_width  = 400
    snap_height = 300

    # Select evenly spaced generation indices (always include the last)
    indices = if n_snap == 1
                [n_gen - 1]
              else
                (0...n_snap).map { |i| ((i * (n_gen - 1).to_f) / (n_snap - 1)).round }
              end.uniq

    # Generate each snapshot as a gruff Bar chart saved to a temp file
    panels = indices.map do |gi|
      gen  = gi + 1
      ps   = @gene_p_history[gi]
      bins = Array.new(N_BINS, 0)
      ps.each { |p| bins[bin_index(p)] += 1 }

      labels = N_BINS.times.each_with_object({}) do |i, h|
        h[i] = format("%.1f", i * BIN_WIDTH) if i % 5 == 0
      end

      g = Gruff::Bar.new(snap_width)
      g.title           = "Gen #{gen}"
      g.title_font_size = 16
      g.labels          = labels
      g.minimum_value   = 0
      g.maximum_value   = @config.n
      g.hide_legend     = true
      g.margins         = 10
      g.data("gen #{gen}", bins)

      tmp = File.join(dir, ".snap_#{gi}.png")
      g.write(tmp)
      tmp
    end

    # Stitch panels horizontally with RMagick
    images   = panels.map { |f| Magick::Image.read(f).first }
    total_w  = snap_width * images.size
    combined = Magick::Image.new(total_w, snap_height)
    combined.background_color = "white"
    images.each_with_index do |img, i|
      combined.composite!(img, i * snap_width, 0, Magick::OverCompositeOp)
    end

    # Add overall title with 30px top margin
    combined = combined.extent(total_w, snap_height + 30, 0, -30)
    tx = Magick::Draw.new
    tx.gravity(Magick::NorthGravity)
    tx.fill("black")
    tx.font("/System/Library/Fonts/Helvetica.ttc")
    tx.font_size(14)
    tx.text(0, 6, "gene_p Snapshots  (ESS = #{format('%.3f', @config.ess)})")
    tx.draw(combined)

    combined.write(File.join(dir, "gene_p_snapshots.png"))
    panels.each { |f| File.delete(f) rescue nil }
  end

  end


class Simulator
  def initialize(config = Config.default)
    @config      = config
    @population  = Population.new(config)
    @reporter    = Reporter.new(config)
    @visualizer  = config.viz_dir ? Visualizer.new(config) : nil
  end

  def run
    @reporter.print_header
 
    @config.generations.times do |i|
      gen = i + 1
      @population.step
      stats = @population.stats
      @reporter.report(gen, stats, @population.last_k, @population.last_early_stop)
      @visualizer&.record(gen, @population.individuals.map(&:gene_p),
                          @population.last_k)
    end
 
    @reporter.print_footer(@population.stats)
    @visualizer&.render
  end
end


# CLI: simple command-line argument parsing
def parse_args(argv, config)
  i = 0
  while i < argv.size
    case argv[i]
    when '--generations'  then config.generations      = argv[i + 1].to_i;   i += 2
    when '--n'            then config.n                = argv[i + 1].to_i;   i += 2
    when '--v'            then config.v                = argv[i + 1].to_f;   i += 2
    when '--c'            then config.c                = argv[i + 1].to_f;   i += 2
    when '--mu'           then config.mu               = argv[i + 1].to_f;   i += 2
    when '--mut-sigma'    then config.mut_sigma        = argv[i + 1].to_f;   i += 2
    when '--max-battles'  then config.max_battles      = argv[i + 1].to_i;   i += 2
    when '--k-threshold'  then config.k_threshold      = argv[i + 1].to_i;   i += 2
    when '--log-interval' then config.log_interval     = argv[i + 1].to_i;   i += 2
    when '--no-csv'       then config.csv_path         = nil;                 i += 1
    when '--csv'          then config.csv_path         = argv[i + 1];         i += 2
    when '--no-viz'       then config.viz_dir          = nil;                 i += 1
    when '--viz-dir'      then config.viz_dir          = argv[i + 1];         i += 2
    when '--viz-snapshots' then config.viz_snapshots     = argv[i + 1].to_i;   i += 2
    else                  i += 1
    end
  end
  config
end

# Entry point
config = parse_args(ARGV, Config.default)
Simulator.new(config).run
