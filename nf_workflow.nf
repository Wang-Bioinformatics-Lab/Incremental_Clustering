#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.input_spectra = "/home/user/research/Incremental_Clustering/data"

//This publish dir is mostly  useful when we want to import modules in other workflows, keep it here usually don't change it
params.publishdir = "$baseDir/nf_output"
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




process CLUSTERING {
    publishDir "$params.publishdir", mode: 'copy'
    conda "$baseDir/bin/conda_env.yml"

    input:
    path input

    output:
    file "results/*mzML" optional true
    file "results/*tsv" optional true
    file "results/*csv" optional true
    file "results/*feather" optional true
    file "results/*parquet" optional true

    script:
    """
    mkdir results
    python3 $TOOL_FOLDER/incremental_clustering.py \
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

    CLUSTERING(params.input_spectra)

    // Alternatively we can put everyhthing in the main from the above right here
}
