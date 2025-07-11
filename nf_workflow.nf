#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.input_spectra = "./data"

//This publish dir is mostly  useful when we want to import modules in other workflows, keep it here usually don't change it

params.publishdir = "$launchDir"
TOOL_FOLDER = "$baseDir/bin"
params.output_dir = "$baseDir/results"
params.checkpoint_dir  = "$baseDir/checkpoint"

// Falcon parameters with defaults
params.precursor_tol = "20 ppm"
params.fragment_tol = 0.05
params.min_mz_range = 0
params.min_mz = 0
params.max_mz = 30000
params.eps = 0.1

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


process CLUSTERING {
    publishDir "$_publishdir", mode: 'copy'
    conda "$baseDir/bin/conda_env.yml"

    input:
    path input

    output:
    file "results/*mzML" optional true
    file "results/*tsv" optional true
    file "results/*csv" optional true
    file "results/*feather" optional true
    file "results/*parquet" optional true
    file "results/*txt" optional true
    file "results/*db" optional true
    file "results/*bin" optional true

    // This is necessary because the glibc libraries are not always used in the conda environment, and defaults to the system which could be old
    beforeScript 'export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$CONDA_PREFIX/lib'

    script:
    """
    mkdir results

    python3 $TOOL_FOLDER/incremental_clustering_sep_ver.py \
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

// TODO: This main will define the workflow that can then be imported, not really used right now, but can be


workflow {

    CLUSTERING(Channel.fromPath(params.input_spectra))

    // Alternatively we can put everyhthing in the main from the above right here
}
