# SEU_CNII - 13

## 1 Clone our repo

`git clone [https://github.com/yindaheng98/NTIRE2023_ESR](https://github.com/yindaheng98/NTIRE2023_ESR)`

Our model path is: `NTIRE2023_ESR/model_zoo/PRFDN_28.992.pth`

Our network path is `NTIRE2023_ESR/models/rfdn_half`

## 2 Test our code

Execute the following script to set the ROOT variable in the shell to the project path(e.g. `ROOT=/home/seu/NTIRE2023_ESR`) and create the directory `results` to save the results.

```bash
ROOT=PATH_TO_OUR_PROJECT
rm -rf "$ROOT/results"
mkdir -p "$ROOT/results"
```

You can select the GPU by setting CUDA_VISIBLE_DEVICES=[GPU_id](e.g `CUDA_VISIBLE_DEVICES=0`). 

Then you can execute our test code as follow:

```bash
python test_demo.py \
  --data_dir PATH_TO_DATASET \
  --save_dir "$ROOT/results" \
  --model_id 13
```

The directory structure of the dataset of **PATH_TO_DATASET** is the same as that of the official repository.  Our model id is **13**.

The results will be stored in **ROOT/results.json** and **ROOT/results.txt.**

To check the results you can execute the following script:

```bash
printf "%20s %12s %17s %14s %5s\n" model_name valid_memory valid_ave_runtime valid_ave_psnr flops
for line in $(cat results.json | jq -r 'to_entries|.[]|[.key,.value.valid_memory,.value.valid_ave_runtime,.value.valid_ave_psnr,.value.flops|tostring] | join(",")'); do
  printf "%20s %12f %17f %14f %5f\n" $(echo $line | sed 's/,/ /g')
done
```

The results then will be displayed.



——————————————————————————————————————————————————————————————————————————



# [NTIRE 2023 Challenge on Efficient Super-Resolution](https://cvlai.net/ntire/2023/) @ [CVPR 2023](https://cvpr2023.thecvf.com/)

## How to test the baseline model?

1. `git clone https://github.com/ofsoundof/NTIRE2023_ESR.git`
2. Select the model you would like to test from [`run.sh`](./run.sh)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python test_demo.py --data_dir [path to your data dir] --save_dir [path to your save dir] --model_id 0
    ```
    - Be sure the change the directories `--data_dir` and `--save_dir`.
   
## How to add your model to this baseline?
1. Register your team in the [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1oekPThh5mq9qKax0hPZiQSHlqTjaoQa-IBfrQkwN7gk/edit?usp=sharing) and get your team ID.
2. Put your the code of your model in `./models/[Your_Team_ID]_[Your_Model_Name].py`
   - Please add **only one** file in the folder `./models`. **Please do not add other submodules**.
   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02 
3. Put the pretrained model in `./model_zoo/[Your_Team_ID]_[Your_Model_Name].[pth or pt or ckpt]`
   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02  
4. Add your model to the model loader `./test_demo/select_model` as follows:
    ```python
        elif model_id == [Your_Team_ID]:
            # define your model and load the checkpoint
    ```
   - Note: Please set the correct data_range, either 255.0 or 1.0
5. Send us the command to download your code, e.g, 
   - `git clone [Your repository link]`
   - We will do the following steps to add your code and model checkpoint to the repository.
   
## How to calculate the number of parameters, FLOPs, and activations

```python
    from utils.model_summary import get_model_flops, get_model_activation
    from models.team00_RFDN import RFDN
    model = RFDN()
    
    input_dim = (3, 256, 256)  # set the input dimension
    activations, num_conv = get_model_activation(model, input_dim)
    activations = activations / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
    print("{:>16s} : {:<d}".format("#Conv2d", num_conv))

    flops = get_model_flops(model, input_dim, False)
    flops = flops / 10 ** 9
    print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    num_parameters = num_parameters / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
```

## License and Acknowledgement
This code repository is release under [MIT License](LICENSE). 
