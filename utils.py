from collections import Counter

def extract_region_name(full_name):
    """Extract anatomical region name from the full voxel name."""
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
    """
    Compute a master region list from all protocols and stages for consistent ordering.
    """
    from collections import Counter
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
    Returns a Counter mapping region -> number of waves that had that region as an origin.
    """
    region_counts = Counter()
    for result in results:
        if 'origins' in result and not result['origins'].empty:
            wave_origin_regions = set(result['origins']['region'].tolist())
            region_counts.update(wave_origin_regions)
    return region_counts


def calculate_involvement_statistics(involvement_data):
    """
    Calculate mean, median, std, count for a list of involvement percentages.
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
    Collect wave-level data across all protocols (for single-group usage).
    """
    consolidated_data = {'pre': [], 'early': [], 'late': [], 'post': []}
    for protocol in results_by_protocol:
        for stage in ['pre', 'early', 'late', 'post']:
            if stage in results_by_protocol[protocol]:
                consolidated_data[stage].extend(results_by_protocol[protocol][stage])
    return consolidated_data


def collect_data_by_treatment_group(results_by_treatment_group):
    """
    Collect wave-level data across all protocols for each treatment group.
    """
    consolidated_data = {}
    for group in results_by_treatment_group:
        consolidated_data[group] = {'pre': [], 'early': [], 'late': [], 'post': []}
        for protocol in results_by_treatment_group[group]:
            for stage in ['pre', 'early', 'late', 'post']:
                if stage in results_by_treatment_group[group][protocol]:
                    consolidated_data[group][stage].extend(results_by_treatment_group[group][protocol][stage])
    return consolidated_data


def read_subject_condition_mapping(json_path):
    """
    Read the Subject_Condition JSON file and return a mapping of subject IDs to conditions.
    
    Args:
        json_path: Path to the JSON file containing subject-condition mappings
        
    Returns:
        Dictionary mapping subject IDs to conditions (Active/SHAM) or None if error
    """
    import json
    try:
        with open(json_path, 'r') as f:
            mapping = json.load(f)
        return mapping
    except Exception as e:
        print(f"Error reading Subject_Condition JSON file: {str(e)}")
        return None


def scan_available_subjects_and_nights(directory_path):
    """
    Scan the directory for available subjects and nights.
    
    Args:
        directory_path: Path to the EEG data directory
        
    Returns:
        Tuple of (subjects, nights) where each is a sorted list of available options
    """
    from pathlib import Path
    
    subjects = []
    nights = set()
    
    dir_path = Path(directory_path)
    subject_dirs = [d for d in dir_path.iterdir() if d.is_dir() and d.name.startswith("Subject_")]
    
    for subject_dir in subject_dirs:
        subjects.append(subject_dir.name)
        night_dirs = [d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith("Night")]
        for night_dir in night_dirs:
            nights.add(night_dir.name)
    
    return sorted(subjects), sorted(list(nights))
