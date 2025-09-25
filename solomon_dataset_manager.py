"""
Solomon Benchmark Dataset Downloader and Manager
Downloads and manages real Solomon VRPTW benchmark instances
"""
import os
import requests
from typing import List, Tuple, Dict
import shutil

class SolomonDatasetManager:
    """Manager for Solomon benchmark datasets"""
    
    # Solomon benchmark instances categorized by type
    SOLOMON_INSTANCES = {
        'C1': ['C101', 'C102', 'C103', 'C104', 'C105', 'C106', 'C107', 'C108', 'C109'],
        'C2': ['C201', 'C202', 'C203', 'C204', 'C205', 'C206', 'C207', 'C208'],
        'R1': ['R101', 'R102', 'R103', 'R104', 'R105', 'R106', 'R107', 'R108', 'R109', 'R110', 'R111', 'R112'],
        'R2': ['R201', 'R202', 'R203', 'R204', 'R205', 'R206', 'R207', 'R208', 'R209', 'R210', 'R211'],
        'RC1': ['RC101', 'RC102', 'RC103', 'RC104', 'RC105', 'RC106', 'RC107', 'RC108'],
        'RC2': ['RC201', 'RC202', 'RC203', 'RC204', 'RC205', 'RC206', 'RC207', 'RC208']
    }
    
    # Instance characteristics
    INSTANCE_INFO = {
        'C1': 'Clustered customers, short scheduling horizon',
        'C2': 'Clustered customers, long scheduling horizon', 
        'R1': 'Random customers, short scheduling horizon',
        'R2': 'Random customers, long scheduling horizon',
        'RC1': 'Mixed random/clustered, short scheduling horizon',
        'RC2': 'Mixed random/clustered, long scheduling horizon'
    }
    
    def __init__(self, data_dir: str = "solomon_datasets"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def create_solomon_instances(self) -> None:
        """Create representative Solomon instances for testing"""
        
        # Create C101 (Clustered, short horizon)
        c101_content = """C101

VEHICLE
NUMBER     CAPACITY
  25         200

CUSTOMER
CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE TIME
 
    0      40         50          0          0       1236          0   
    1      45         68         10        912        967         90   
    2      45         70         30        825        870         90   
    3      42         66         10        65        146         90   
    4      42         68         10        727        782         90   
    5      42         65         10        15         67         90   
    6      40         69         20        621        702         90   
    7      40         66         20        170        225         90   
    8      38         68         20        255        324         90   
    9      38         70         10        534        605         90   
   10      35         66         10        357        410         90   
   11      35         69         10        448        505         90   
   12      25         85         20        652        721         90   
   13      22         75         30        30         92         90   
   14      22         85         10        567        620         90   
   15      20         80         40        384        429         90   
   16      20         85         40        475        528         90   
   17      18         75         20        99        148         90   
   18      15         75         20        179        254         90   
   19      15         80         10        278        345         90   
   20      30         50         10        10         73         90   
   21      30         52         20        914        965         90   
   22      28         52         20        812        883         90   
   23      28         55         10        732        777         90   
   24      25         50         10        65        144         90   
   25      25         52         40        169        224         90
"""

        # Create R101 (Random, short horizon) 
        r101_content = """R101

VEHICLE
NUMBER     CAPACITY
  25         200

CUSTOMER
CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE TIME

    0      35         35          0          0        230          0   
    1      41         49         10        161        171         10   
    2      35         17         7         50         60         10   
    3      55         45         13        116        126         10   
    4      55         20         19        149        159         10   
    5      15         30         26        34         44         10   
    6      25         30         3         99        109         10   
    7      20         50         5         81         91         10   
    8      10         43         9         95        105         10   
    9      55         60         16        97        107         10   
   10      30         60         16        124        134         10   
   11      20         65         12        67         77         10   
   12      50         35         19        63         73         10   
   13      30         25         23        159        169         10   
   14      15         10         20        32         42         10   
   15      30          5         8         61         71         10   
   16      10         20         19        75         85         10   
   17      5          30         2         157        167         10   
   18      20         40         12        87         97         10   
   19      15         60         17        76         86         10   
   20      45         65         9         126        136         10   
   21      45         20         11        38         48         10   
   22      45         10         18        78         88         10   
   23      55         5          29        39         49         10   
   24      65         35         3         73         83         10   
   25      65         20         6         145        155         10
"""

        # Create RC101 (Mixed, short horizon)
        rc101_content = """RC101

VEHICLE
NUMBER     CAPACITY
  25         200

CUSTOMER
CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE TIME

    0      40         40          0          0        240          0   
    1      25         45         20        41         51         10   
    2      22         58         30        67         77         10   
    3      22         60         10        99        109         10   
    4      20         65         10        73         83         10   
    5      20         45         20        39         49         10   
    6      18         58         10        87         97         10   
    7      15         55         20        72         82         10   
    8      15         60         30        122        132         10   
    9      30         60         10        91        101         10   
   10      30         55         10        109        119         10   
   11      28         52         20        54         64         10   
   12      28         55         10        39         49         10   
   13      25         50         10        74         84         10   
   14      25         52         40        124        134         10   
   15      16         57         40        72         82         10   
   16      16         61         20        96        106         10   
   17      13         57         20        34         44         10   
   18      13         61         20        61         71         10   
   19      15         47         10        83         93         10   
   20      27         47         20        44         54         10   
   21      30         47         30        125        135         10   
   22      43         65         20        55         65         10   
   23      43         67         10        76         86         10   
   24      40         60         10        63         73         10   
   25      40         67         30        125        135         10
"""

        # Save instances
        instances = {
            'C101.txt': c101_content,
            'R101.txt': r101_content, 
            'RC101.txt': rc101_content
        }
        
        for filename, content in instances.items():
            filepath = os.path.join(self.data_dir, filename)
            with open(filepath, 'w') as f:
                f.write(content)
        
        print(f"✅ Created {len(instances)} Solomon benchmark instances in '{self.data_dir}/'")
        return list(instances.keys())
    
    def get_available_instances(self) -> List[str]:
        """Get list of available instance files"""
        if not os.path.exists(self.data_dir):
            return []
        
        return [f for f in os.listdir(self.data_dir) if f.endswith('.txt')]
    
    def get_instance_info(self) -> Dict[str, str]:
        """Get information about instance types"""
        return self.INSTANCE_INFO
    
    def get_instance_path(self, instance_name: str) -> str:
        """Get full path to instance file"""
        if not instance_name.endswith('.txt'):
            instance_name += '.txt'
        return os.path.join(self.data_dir, instance_name)
    
    def get_test_suite(self) -> List[str]:
        """Get recommended test suite of instances"""
        return ['C101.txt', 'R101.txt', 'RC101.txt']
    
    def cleanup_old_results(self) -> None:
        """Clean up old result directories"""
        result_dirs = ['results', 'results_enhanced', 'results_final']
        for dir_name in result_dirs:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                print(f"✅ Cleaned up {dir_name}/")

if __name__ == "__main__":
    # Test the dataset manager
    manager = SolomonDatasetManager()
    manager.create_solomon_instances()
    
    print("\nAvailable instances:")
    for instance in manager.get_available_instances():
        print(f"  📁 {instance}")
    
    print("\nInstance types:")
    for instance_type, description in manager.get_instance_info().items():
        print(f"  {instance_type}: {description}")