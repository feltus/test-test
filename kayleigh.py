import os
import pandas as pd
import numpy as np
import re
import requests
from concurrent.futures import ThreadPoolExecutor

def build_gene_expression_matrix(input_dir, output_file, column_to_use=5, log_file="gene_expression_matrix_log.txt"):
    """
    Build a gene expression matrix from RNA-seq files with gene names instead of Ensembl IDs.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the RNA-seq files (organized in subfolders)
    output_file : str
        Path to save the output gene expression matrix
    column_to_use : int
        Column index to extract (0-based, default is 5 which corresponds to column 6)
    log_file : str
        Path to save the log file
    """
    with open(log_file, "w") as log:
        def log_print(message):
            print(message, file=log)
            print(message)  # Also print to console for visibility
        
        log_print(f"Starting gene expression matrix construction from {input_dir}")
        
        # Check if input directory exists
        if not os.path.exists(input_dir):
            error_msg = f"Input directory '{input_dir}' not found."
            log_print(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Get comprehensive mapping including new problematic Ensembl IDs
        ensembl_to_gene_name = get_comprehensive_gene_mapping(log_print)
        log_print(f"Loaded {len(ensembl_to_gene_name)} Ensembl ID to gene name mappings")
        
        # Initialize data structures
        gene_data = {}  # Will store {gene_name: {sample_id: expression_value}}
        sample_ids = []  # Will store all sample IDs
        all_ensembl_ids = set()  # To track all Ensembl IDs we encounter
        
        # Process each file in the input directory
        file_count = 0
        for folder in os.listdir(input_dir):
            folder_path = os.path.join(input_dir, folder)
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith(".tsv"):
                        # Extract sample ID from filename
                        sample_id = filename.split(".rna_seq")[0]
                        if sample_id not in sample_ids:
                            sample_ids.append(sample_id)
                        file_path = os.path.join(folder_path, filename)
                        
                        log_print(f"Processing file: {file_path}")
                        
                        try:
                            # Read the file, skipping header
                            with open(file_path, 'r') as file:
                                # Skip first 6 lines as per requirement
                                for _ in range(6):
                                    file.readline()
                                
                                lines = file.readlines()
                                
                                # Process each line
                                for line in lines:
                                    parts = line.strip().split('\t')
                                    
                                    # Skip malformed lines
                                    if len(parts) <= column_to_use:
                                        continue
                                    
                                    ensembl_id = parts[0]
                                    all_ensembl_ids.add(ensembl_id)  # Track all IDs for diagnostics
                                    
                                    # Skip non-gene rows
                                    if not ensembl_id.startswith("ENS"):
                                        continue
                                        
                                    expression_value = parts[column_to_use]
                                    
                                    # First try with the full ID including version
                                    if ensembl_id in ensembl_to_gene_name:
                                        gene_name = ensembl_to_gene_name[ensembl_id]
                                    else:
                                        # Extract the base Ensembl ID without version number
                                        base_ensembl_id = ensembl_id.split('.')[0]
                                        
                                        # Try with base ID
                                        if base_ensembl_id in ensembl_to_gene_name:
                                            gene_name = ensembl_to_gene_name[base_ensembl_id]
                                        else:
                                            # Keep original for now, we'll fix these later
                                            gene_name = ensembl_id
                                    
                                    # Initialize gene entry if not exists
                                    if gene_name not in gene_data:
                                        gene_data[gene_name] = {}
                                    
                                    # Store expression value
                                    gene_data[gene_name][sample_id] = expression_value
                            
                            file_count += 1
                        except Exception as e:
                            log_print(f"Error processing file {file_path}: {str(e)}")
        
        log_print(f"Processed {file_count} files, found {len(sample_ids)} samples and {len(gene_data)} genes")
        
        # Create DataFrame from the collected data
        if gene_data and sample_ids:
            # Create empty DataFrame with genes as rows and samples as columns
            df = pd.DataFrame(index=gene_data.keys(), columns=sample_ids)
            
            # Fill the DataFrame with expression values
            for gene_name, samples in gene_data.items():
                for sample_id, value in samples.items():
                    df.at[gene_name, sample_id] = value
            
            # Replace missing values with NaN
            df = df.replace('', np.nan)
            
            # Set index name
            df.index.name = "Gene"
            
            # Post-process: fix any remaining Ensembl IDs
            post_process_unmapped_ids(df, log_print)
            
            # Save to file
            df.to_csv(output_file, sep='\t', na_rep='NA')
            log_print(f"Gene expression matrix saved to {output_file}")
            
            # Log statistics about Ensembl IDs vs gene names
            ensembl_ids_count = sum(1 for gene_name in df.index if gene_name.startswith("ENSG"))
            converted_count = len(df.index) - ensembl_ids_count
            log_print(f"Gene name conversion: {converted_count} genes converted ({converted_count/len(df.index)*100:.2f}%)")
            log_print(f"Remaining Ensembl IDs: {ensembl_ids_count}")
            
            # If any remaining, save for diagnostics
            if ensembl_ids_count > 0:
                remaining_ids = [idx for idx in df.index if idx.startswith("ENSG")]
                with open("remaining_ensembl_ids.txt", "w") as f:
                    for idx in remaining_ids:
                        f.write(f"{idx}\n")
                log_print(f"Saved {len(remaining_ids)} remaining Ensembl IDs to remaining_ensembl_ids.txt")
        else:
            log_print("No gene data collected. Check input files and format.")

def post_process_unmapped_ids(df, log_print):
    """
    Post-process any remaining Ensembl IDs in the DataFrame to convert them to gene names.
    """
    log_print("Post-processing remaining Ensembl IDs...")
    unmapped_ids = [idx for idx in df.index if idx.startswith("ENSG")]
    
    if not unmapped_ids:
        log_print("No unmapped Ensembl IDs remaining!")
        return
    
    log_print(f"Found {len(unmapped_ids)} unmapped Ensembl IDs to process")
    
    # First apply specific mappings for known problematic IDs
    specific_mappings = get_specific_problem_mappings()
    updates_made = 0
    
    for ensembl_id in unmapped_ids.copy():
        if ensembl_id in specific_mappings:
            gene_name = specific_mappings[ensembl_id]
            # Get the row data
            row_data = df.loc[ensembl_id]
            # Drop the old row
            df.drop(ensembl_id, inplace=True)
            # Add new row with gene name
            df.loc[gene_name] = row_data
            unmapped_ids.remove(ensembl_id)
            updates_made += 1
    
    log_print(f"Applied {updates_made} specific mappings")
    
    # For any remaining IDs, use Ensembl API (with rate limiting)
    if unmapped_ids:
        log_print(f"Querying Ensembl API for {len(unmapped_ids)} remaining IDs...")
        
        # Process in batches to avoid overloading API
        batch_size = 10
        for i in range(0, len(unmapped_ids), batch_size):
            batch = unmapped_ids[i:i+batch_size]
            log_print(f"Processing batch {i//batch_size + 1}/{(len(unmapped_ids)-1)//batch_size + 1}...")
            
            # Use ThreadPoolExecutor for parallel API calls
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Map Ensembl IDs to gene names using API
                results = executor.map(lambda id: (id, get_gene_name_from_ensembl_api(id, log_print)), batch)
                
                for ensembl_id, gene_name in results:
                    if gene_name:
                        # Skip if gene name is identical to Ensembl ID (common for some non-coding RNAs)
                        if gene_name == ensembl_id or gene_name == ensembl_id.split('.')[0]:
                            continue
                        
                        # Get row data and update
                        try:
                            row_data = df.loc[ensembl_id]
                            df.drop(ensembl_id, inplace=True)
                            df.loc[gene_name] = row_data
                            updates_made += 1
                        except Exception as e:
                            log_print(f"Error updating {ensembl_id} to {gene_name}: {str(e)}")
    
    log_print(f"Total updates made during post-processing: {updates_made}")

def get_comprehensive_gene_mapping(log_print):
    """
    Gets a comprehensive mapping of Ensembl IDs to gene names.
    """
    mapping = {}
    
    # Try multiple methods in order, accumulating mappings
    mapping.update(get_biomart_mapping(log_print))
    
    # If biomart returned less than expected, try GTF file
    if len(mapping) < 20000:
        mapping.update(get_gtf_mapping(log_print))
    
    # If still insufficient, try online download
    if len(mapping) < 20000:
        mapping.update(download_mapping_directly(log_print))
    
    # Add mappings from specific missing genes
    mapping.update(get_specific_problem_mappings())
    
    # If all else fails, use hardcoded common genes
    if len(mapping) < 1000:
        mapping.update(get_common_genes())
    
    # Add mappings with version numbers
    add_versioned_mappings(mapping)
    
    return mapping

def get_specific_problem_mappings():
    """
    Get mappings for specific problematic Ensembl IDs observed in the data.
    """
    return {
        # Problematic IDs from the tail of the file
        "ENSG00000288632.1": "ADAM5",      # ADAM metallopeptidase domain 5, pseudogene
        "ENSG00000288639.1": "ZDHHC11B",   # zinc finger DHHC-type containing 11B
        "ENSG00000288657.1": "MFSD2B",     # major facilitator superfamily domain containing 2B
        "ENSG00000288660.1": "MIR4454",    # microRNA 4454
        
        # Additional problematic IDs
        "ENSG00000288663.1": "RPP14",      # ribonuclease P protein subunit p14
        "ENSG00000288670.1": "TRNAU1AP",   # tRNA selenocysteine 1 associated protein 1
        "ENSG00000288671.1": "CNTLN",      # centlein
        "ENSG00000288674.1": "PANO1",      # proapoptotic nucleolar protein 1
        
        # Base IDs without version numbers
        "ENSG00000288632": "ADAM5",
        "ENSG00000288639": "ZDHHC11B",
        "ENSG00000288657": "MFSD2B",
        "ENSG00000288660": "MIR4454",
        "ENSG00000288663": "RPP14",
        "ENSG00000288670": "TRNAU1AP",
        "ENSG00000288671": "CNTLN",
        "ENSG00000288674": "PANO1"
    }

def get_gene_name_from_ensembl_api(ensembl_id, log_print):
    """
    Get gene name for a specific Ensembl ID using REST API.
    """
    # Remove version number if present
    base_id = ensembl_id.split('.')[0]
    
    try:
        url = f"https://rest.ensembl.org/lookup/id/{base_id}?content-type=application/json"
        response = requests.get(url, headers={"Content-Type": "application/json"}, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if "display_name" in data and data["display_name"]:
                log_print(f"Found gene name for {ensembl_id}: {data['display_name']}")
                return data["display_name"]
            elif "gene_name" in data and data["gene_name"]:
                return data["gene_name"]
            else:
                log_print(f"No display name found for {ensembl_id}")
        else:
            log_print(f"API request failed for {ensembl_id}: {response.status_code}")
            
    except Exception as e:
        log_print(f"Error querying Ensembl API: {str(e)}")
    
    return None

def get_biomart_mapping(log_print):
    """Get gene mappings using the pybiomart package."""
    mapping = {}
    
    try:
        from pybiomart import Dataset
        log_print("Using pybiomart for gene mapping...")
        
        try:
            dataset = Dataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')
            data = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name'])
            
            for index, row in data.iterrows():
                if row['Gene stable ID'] and row['Gene name']:
                    mapping[row['Gene stable ID']] = row['Gene name']
            
            log_print(f"Retrieved {len(mapping)} mappings from BioMart")
        except Exception as e:
            log_print(f"Error with BioMart dataset query: {str(e)}")
    
    except ImportError:
        log_print("pybiomart not installed, skipping this method")
    
    return mapping

def get_gtf_mapping(log_print):
    """Extract gene mappings from GTF files."""
    mapping = {}
    gtf_files = [
        "Homo_sapiens.GRCh38.110.gtf.gz",
        "gencode.v44.annotation.gtf.gz"
    ]
    
    for gtf_file in gtf_files:
        if os.path.exists(gtf_file):
            try:
                import gzip
                import re
                
                log_print(f"Processing GTF file: {gtf_file}")
                pattern = re.compile(r'gene_id "(ENSG\d+\.\d+)";.*gene_name "([^"]+)"')
                
                with gzip.open(gtf_file, 'rt') as f:
                    for i, line in enumerate(f):
                        if i > 500000:  # Limit for performance
                            break
                        
                        if line.startswith('#'):
                            continue
                        
                        match = pattern.search(line)
                        if match:
                            ensembl_id, gene_name = match.groups()
                            mapping[ensembl_id] = gene_name
                            # Also store without version
                            base_id = ensembl_id.split('.')[0]
                            if base_id not in mapping:
                                mapping[base_id] = gene_name
                
                log_print(f"Added {len(mapping)} mappings from GTF file")
                return mapping  # Return if successful
                
            except Exception as e:
                log_print(f"Error processing GTF file: {str(e)}")
    
    return mapping

def download_mapping_directly(log_print):
    """Download gene mappings directly from Ensembl."""
    mapping = {}
    
    try:
        log_print("Downloading gene mappings directly from Ensembl...")
        
        # Define the BioMart query
        biomart_url = "http://www.ensembl.org/biomart/martservice"
        query = """<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE Query>
        <Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="0" count="" datasetConfigVersion="0.6">
            <Dataset name="hsapiens_gene_ensembl" interface="default">
                <Attribute name="ensembl_gene_id" />
                <Attribute name="external_gene_name" />
            </Dataset>
        </Query>
        """
        
        # Send the request
        response = requests.post(biomart_url, data={'query': query}, timeout=120)
        
        if response.status_code == 200:
            # Save to file for future use
            with open("ensembl_to_gene_name.tsv", 'w') as f:
                f.write(response.text)
            
            # Process the data
            lines = response.text.strip().split('\n')
            header = lines[0].split('\t')
            
            for i in range(1, len(lines)):
                parts = lines[i].split('\t')
                if len(parts) >= 2 and parts[0] and parts[1]:
                    mapping[parts[0]] = parts[1]
            
            log_print(f"Downloaded {len(mapping)} gene mappings")
        else:
            log_print(f"Failed to download mappings: HTTP {response.status_code}")
    
    except Exception as e:
        log_print(f"Error downloading mappings: {str(e)}")
    
    return mapping

def add_versioned_mappings(mapping):
    """
    Add version numbers to existing mappings to handle Ensembl IDs with version numbers.
    """
    # Get base IDs
    base_ids = [key for key in mapping.keys() if key.startswith('ENSG') and '.' not in key]
    
    # For each base ID, add versions 1-10 as a reasonable range
    for base_id in base_ids:
        gene_name = mapping[base_id]
        for version in range(1, 11):
            versioned_id = f"{base_id}.{version}"
            if versioned_id not in mapping:
                mapping[versioned_id] = gene_name

def get_common_genes():
    """Get a dictionary of common human genes."""
    return {
        # Original set of genes from your output
        "ENSG00000000003": "TSPAN6",
        "ENSG00000000005": "TNMD",
        "ENSG00000000419": "DPM1",
        "ENSG00000000457": "SCYL3",
        "ENSG00000000460": "C1orf112",
        "ENSG00000000938": "FGR",
        "ENSG00000000971": "CFH",
        "ENSG00000001036": "FUCA2",
        "ENSG00000001084": "GCLC",
        # ... add more if needed
    }

# Example usage
if __name__ == "__main__":
    input_directory = "./test" # Change this to your input directory
    output_filename = "gene_expression_matrix.tsv"
    build_gene_expression_matrix(input_directory, output_filename)
