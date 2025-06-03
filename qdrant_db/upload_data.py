from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchText
from qdrant_client.http.exceptions import UnexpectedResponse

from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import uuid

# Load environment variables from .env file
load_dotenv()

# Get Qdrant credentials from environment variables
api_key = os.getenv("QDRANT_API_KEY")
cluster_url = os.getenv("QDRANT_CLUSTER_URL")

# Connect to Qdrant cluster if credentials are available, otherwise use local
if api_key and cluster_url:
    client = QdrantClient(url=cluster_url, api_key=api_key)
    print(f"Connected to Qdrant cloud cluster at {cluster_url}")
else:
    client = QdrantClient(url="http://localhost:6333")
    print("Connected to local Qdrant instance")

colName = "OmiyDB"

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

model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
model = SentenceTransformer(model_name)
print(f"Model '{model_name}' loaded successfully!")

# Convert descriptions to embeddings and upload to Qdrant
tool_ids = [] # e.g. 0a1ab736-8765-4900-827b-7dc6ebfd8f2b
tool_vectors = [] # e.g. [-0.04451117664575577, 0.07700273394584656, ..., 0.3271920084953308]
tool_payloads = [] # e.g. {'tool_name': 'BioPython', 'description': 'A set of freely ... for developers.', 'url': 'https://biopython.org/'}

# Process each tool
for tool in bioinformatics_tools:
    tool_name = tool['tool_name']
    
    # Check if the tool already exists in the collection
    try:
        existing_tools = client.scroll(
            collection_name=colName,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="tool_name",
                        match=MatchText(text=tool_name)
                    )
                ]
            ),
            limit=1
        )[0]
        
        if existing_tools:
            print(f"Tool '{tool_name}' already exists in the collection. Skipping.")
            continue
    except UnexpectedResponse:
        # Collection might not exist yet or other API issue
        pass
    
    # Generate a unique ID
    tool_id = str(uuid.uuid4())
    tool_ids.append(tool_id)
    
    # Create vector embedding from the description
    # Combine name and description for better semantic representation
    text_to_embed = f"{tool['tool_name']}: {tool['description']}"
    vector = model.encode(text_to_embed).tolist()
    tool_vectors.append(vector)
    
    # Add payload with all tool data
    tool_payloads.append(tool)

# Only upload if we have tools to upload
if tool_ids:
    client.upsert(
        collection_name=colName,
        points=[
            PointStruct(
                id=tool_id, 
                vector=vector, 
                payload=payload
            )
            for tool_id, vector, payload in zip(tool_ids, tool_vectors, tool_payloads)
        ]
    )
    print(f"Successfully uploaded {len(tool_ids)} new bioinformatics tools to the collection '{colName}'")
else:
    print("No new tools to upload. All tools already exist in the collection.")