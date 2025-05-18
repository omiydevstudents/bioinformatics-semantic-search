from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

# Load environment variables if you have them in a .env file
load_dotenv()

# Same bioinformatics tools data from upload_data.py
bioinformatics_tools = [
    {
        "tool_name": "BioPython",
        "description": "An essential open-source Python library providing a vast collection of modules for biological computation. It enables users to parse bioinformatics file formats (e.g., FASTA, GenBank), manipulate biological sequences, access online databases (like NCBI), perform sequence alignments, conduct phylogenetic analysis, and work with 3D molecular structures.",
        "url": "https://biopython.org/"
    },
    {
        "tool_name": "Bioconductor",
        "description": "An open-source project built on the R statistical programming language, offering a comprehensive suite of R packages for the analysis, comprehension, and visualization of high-throughput genomic data. It is widely used for tasks such as differential gene expression analysis, ChIP-seq, RNA-seq, and handling data from microarrays and next-generation sequencing (NGS).",
        "url": "https://bioconductor.org/"
    },
    {
        "tool_name": "BLAST (Basic Local Alignment Search Tool)",
        "description": "The Basic Local Alignment Search Tool (BLAST) is a cornerstone algorithm and program for comparing primary biological sequence information, such as amino-acid sequences of proteins or nucleotide sequences of DNA/RNA. It rapidly finds regions of local similarity between a query sequence and sequences within a database, enabling researchers to infer functional and evolutionary relationships, identify homologous genes, and annotate newly sequenced genomes.",
        "url": "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
    },
    {
        "tool_name": "Clustal Omega",
        "description": "A widely-used multiple sequence alignment (MSA) program that efficiently aligns three or more protein or nucleic acid (DNA/RNA) sequences. Clustal Omega is known for its speed, accuracy, and ability to handle large datasets, producing biologically meaningful alignments even for divergent sequences. These alignments are crucial for phylogenetic analysis, identifying conserved motifs and domains, and informing protein structure and function prediction.",
        "url": "https://www.ebi.ac.uk/Tools/msa/clustalo/"
    },
    {
        "tool_name": "IGV (Integrative Genomics Viewer)",
        "description": "A high-performance, easy-to-use, interactive desktop tool for the visual exploration of diverse and large-scale genomic datasets. IGV allows users to simultaneously view multiple types of data, such as mapped sequencing reads (e.g., BAM files), genomic variants (e.g., VCF files), gene annotations (e.g., GFF, BED files), and expression levels, facilitating quality control, data interpretation, and discovery in genomics research.",
        "url": "https://software.broadinstitute.org/software/igv/"
    },
    {
        "tool_name": "Galaxy",
        "description": "An open, web-based platform designed for accessible, reproducible, and transparent computational biomedical research. Galaxy enables users, particularly those without extensive programming expertise, to perform complex bioinformatics analyses (e.g., genomics, proteomics, transcriptomics) by providing a user-friendly graphical interface to a vast collection of tools. It allows for the creation, execution, and sharing of analytical workflows, with robust history tracking for enhanced reproducibility.",
        "url": "https://galaxyproject.org/"
    },
    {
        "tool_name": "GATK (Genome Analysis Toolkit)",
        "description": "Developed by the Broad Institute, the Genome Analysis Toolkit (GATK) is a comprehensive software package for analyzing high-throughput sequencing (NGS) data, with a strong focus on variant discovery (SNPs, indels, CNVs, and SVs). It provides a wide array of tools and 'Best Practices' workflows for data pre-processing, germline short variant discovery, somatic variant calling, and quality control, widely used in both research and clinical genomics.",
        "url": "https://gatk.broadinstitute.org/hc/en-us"
    },
    {
        "tool_name": "Cytoscape",
        "description": "An open-source software platform for visualizing, analyzing, and modeling complex biological networks and integrating them with various types of attribute data (e.g., gene expression profiles, clinical data, annotations). Cytoscape enables interactive exploration and analysis of networks such as protein-protein interactions, metabolic pathways, and gene regulatory networks, and is highly extensible through a rich ecosystem of apps, making it a powerful tool for systems biology and network-based data integration.",
        "url": "https://cytoscape.org/"
    }
]

def convert_to_langchain_documents(tools_data):
    """Convert bioinformatics tool data to LangChain Document objects."""
    documents = []
    
    for tool in tools_data:
        # Use the description as the page_content
        content = tool["description"]
        
        # The rest of the data goes into metadata
        metadata = {
            "tool_name": tool["tool_name"],
            "url": tool["url"]
        }
        
        # Create Document object
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    
    return documents

def main():
    # Load the same biomedical BERT model
    embeddings = HuggingFaceEmbeddings(
        model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("Embeddings model loaded successfully!")
    
    # Convert tools to LangChain documents
    documents = convert_to_langchain_documents(bioinformatics_tools)
    print(f"Converted {len(documents)} bioinformatics tools to LangChain documents.")
    
    # Create Qdrant vector store with LangChain
    vector_store = Qdrant.from_documents(
        documents,
        embeddings,
        url="http://localhost:6333",
        collection_name="OmiyDB_LangChain",
        force_recreate=True
    )
    print(f"Successfully created 'OmiyDB_LangChain' collection in Qdrant!")
    
    # Test a simple query
    query = "Which tool provides good workflows for germline short variant discovery from high-throughput sequencing data?"
    results = vector_store.similarity_search(query, k=3)
    
    print("\nSearch results for query:", query)
    for doc in results:
        print(f"Tool: {doc.metadata['tool_name']}")
        print(f"Score: {doc.metadata.get('score', 'N/A')}")
        print(f"Description: {doc.page_content}")
        print(f"URL: {doc.metadata['url']}")
        print("-------------")

if __name__ == "__main__":
    main() 