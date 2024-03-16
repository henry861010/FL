#!/bin/bash

# Navigate to the directory containing your scripts if necessary
# cd /path/to/your/scripts

# Run each Python script sequentially
python main.py ./config/selection_number_pattern/iid/fedavg_1_cifar10_iid.json || echo "------------------1error----------------------------------------"
echo "1. selection_number_pattern/iid/fedavg_1_cifar10_iid scripts have completed."

python main.py ./config/selection_number_pattern/iid/fedavg_all_cifar10_iid.json || echo "------------------2error----------------------------------------"
echo "2. selection_number_pattern/iid/fedavg_all_cifar10_iid scripts have completed."

python main.py ./config/selection_number_pattern/iid/fedavg_uni_cifar10_iid.json || echo "------------------3error----------------------------------------"
echo "3. selection_number_pattern/iid/fedavg_uni_cifar10_iid scripts have completed."

python main.py ./config/selection_number_pattern/iid/fedavg_asc_cifar10_iid.json || echo "------------------4error----------------------------------------"
echo "4. selection_number_pattern/iid/fedavg_asc_cifar10_iid scripts have completed."

python main.py ./config/selection_number_pattern/iid/fedavg_dec_cifar10_iid.json || echo "------------------5error----------------------------------------"
echo "5. selection_number_pattern/iid/fedavg_dec_cifar10_iid scripts have completed."

echo "All scripts have completed."
