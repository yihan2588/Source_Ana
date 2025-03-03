from collections import Counter

def extract_region_name(full_name):
    """Extract anatomical region name from the full voxel name"""
    try:
        if '.' in full_name:
            return full_name.split('.')[0]
        elif "['" in full_name and "." in full_name:
            region = full_name.split("['")[1].split(".")[0]
            return region
        else:
            return full_name
    except:
        return full_name

def compute_master_region_list(results_by_protocol):
    """Compute a master region list from all protocols and stages for consistent ordering"""
    region_frequency = Counter()
    for protocol in results_by_protocol:
        for stage in ['pre', 'early', 'late', 'post']:
            for result in results_by_protocol[protocol][stage]:
                if 'origins' in result and not result['origins'].empty:
                    regions = result['origins']['region'].tolist()
                    region_frequency.update(regions)
    master_region_list = [region for region, _ in region_frequency.most_common()]
    return master_region_list

def calculate_origin_statistics(results, stage, total_waves):
    """
    Calculate origin statistics for a given stage.
    
    Args:
        results: List of result dictionaries for a stage
        stage: Stage name ('pre', 'early', 'late', 'post')
        total_waves: Total number of waves in the stage
        
    Returns:
        Counter object with region counts
    """
    region_counts = Counter()
    for result in results:
        if 'origins' in result and not result['origins'].empty:
            wave_origin_regions = set(result['origins']['region'].tolist())
            region_counts.update(wave_origin_regions)
    return region_counts

def calculate_involvement_statistics(involvement_data):
    """
    Calculate statistics for involvement data.
    
    Args:
        involvement_data: List of involvement percentages
        
    Returns:
        Dictionary with mean, median, std, and count
    """
    import numpy as np
    
    if not involvement_data:
        return {'mean': 0, 'median': 0, 'std': 0, 'count': 0}
    
    return {
        'mean': np.mean(involvement_data),
        'median': np.median(involvement_data),
        'std': np.std(involvement_data),
        'count': len(involvement_data)
    }

def collect_wave_level_data(results_by_protocol):
    """
    Collect wave-level data across all protocols.
    
    Args:
        results_by_protocol: Dictionary with protocols and their results
        
    Returns:
        Dictionary with consolidated data by stage
    """
    consolidated_data = {'pre': [], 'early': [], 'late': [], 'post': []}
    
    for protocol in results_by_protocol:
        for stage in ['pre', 'early', 'late', 'post']:
            if stage in results_by_protocol[protocol]:
                consolidated_data[stage].extend(results_by_protocol[protocol][stage])
    
    return consolidated_data
