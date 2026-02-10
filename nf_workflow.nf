#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.input_spectra = "./data/round1"
params.checkpoint_dir  = "./checkpoint"

// Hyper-Spec / Falcon parameters with defaults
params.precursor_tol = "20 ppm"
params.fragment_tol = 0.05
params.min_mz_range = 0
// Hyper-Spec works best in a reasonable MS/MS range
params.min_mz = 10
params.max_mz = 2000
params.eps = 0.5

// Networking Parameters (set do_networking = "Yes" to run calculatePairs_index / GNPS networking)
params.do_networking = "No"

params.similarity = "gnps"

params.parallelism = 100
params.maxforks = 24

params.min_matched_peaks = 6
params.min_cosine = 0.7
params.max_shift = 1000

// Boiler plate
//This publishdir is mostly  useful when we want to import modules in other workflows, keep it here usually don't change it
params.publishdir = "$launchDir"
TOOL_FOLDER = "$baseDir/bin"
MODULES_FOLDER = "$TOOL_FOLDER/NextflowModules"


// COMPATIBILITY NOTE: The following might be necessary if this workflow is being deployed in a slightly different environemnt
// checking if outdir is defined,
// if so, then set publishdir to outdir
if (params.outdir) {
    _publishdir = params.outdir
}
else{
    _publishdir = params.publishdir
}

// Augmenting with nf_output
_publishdir = "${_publishdir}/nf_output"

// A lot of useful modules are already implemented and added to the nextflow modules, you can import them to use
// the publishdir is a key word that we're using around all our modules to control where the output files will be saved
include {calculatePairs_index} from "$MODULES_FOLDER/nf_networking_modules.nf" addParams(publishdir: _publishdir)
include {prepGNPSParams} from "$MODULES_FOLDER/nf_networking_modules.nf" addParams(publishdir: _publishdir)
include {calculateGNPSPairs} from "$MODULES_FOLDER/nf_networking_modules.nf" addParams(publishdir: _publishdir)
include {calculatePairsEntropy} from "$MODULES_FOLDER/nf_networking_modules.nf" addParams(publishdir: _publishdir)


process CLUSTERING {
    publishDir "$_publishdir", mode: 'copy'

    conda "$TOOL_FOLDER/conda_env.yml"

    input:
    path input

    output:
    file "results/*mzML" optional true
    file "results/*tsv" optional true
    file "results/*csv" optional true
    file "results/*feather" optional true
    file "results/*parquet" optional true
    file "results/*mgf" optional true
    file "results/*txt" optional true
    file "results/*db" optional true
    file "results/*bin" optional true

    // This is necessary because the glibc libraries are not always used in the conda environment, and defaults to the system which could be old
    // Put conda lib directory FIRST to prioritize conda's libstdc++ over system version
    beforeScript 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH'

    script:
    """
    # Ensure conda's libstdc++ is used before system version
    export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH}
    
    mkdir results

    python3 $TOOL_FOLDER/incremental_clustering_hyper_spec.py \
        --folder $input \
        --checkpoint_dir "${params.checkpoint_dir}" \
        --output_dir results \
        --tool_dir $TOOL_FOLDER \
        --precursor_tol "${params.precursor_tol}" \
        --fragment_tol ${params.fragment_tol} \
        --min_mz_range ${params.min_mz_range} \
        --min_mz ${params.min_mz} \
        --max_mz ${params.max_mz} \
        --eps ${params.eps}
    """
}


workflow {

    input_spectra_ch = Channel.fromPath(params.input_spectra)

    // TODO: We should fix this so that relative paths work for the checkpoints

    (_clustered_data_ch, _, _, _, _, _, _, _) = CLUSTERING(input_spectra_ch)

    // Here we do networking
    if (params.do_networking == "Yes") {
        parallelism_val = params.parallelism.toInteger() - 1
        index_parallel_ch = Channel.from(0..parallelism_val)

        enable_peak_filtering = "Yes"

        // Call the networking process
        _index_pairs_ch = calculatePairs_index(_clustered_data_ch.first(), index_parallel_ch, \
        params.fragment_tol, \
        params.min_cosine, \
        params.parallelism, \
        "index_single_charge", \
        enable_peak_filtering)

        _index_pairs_ch.collectFile(name: "networking/merged_pairs.tsv", storeDir: "$_publishdir", keepHeader: true)
    }
}
