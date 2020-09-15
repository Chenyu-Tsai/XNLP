python snli.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 64 --gradient_accumulation_steps 1 --train_batch_size 32 --learning_rate 3e-5 --num_train_epochs 3.0 --output_dir model_output --overwrite_output_dir --eval_all_checkpoints --logging_steps 50 --save_steps 500

python snli_multi_task.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 64 --evaluate_during_training  --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 3.0 --output_dir model_output --overwrite_output_dir --logging_steps 500 --save_steps 1000

{
    'acc_10000': 0.8864052021946759, 'acc_1000': 0.826153220890063, 'acc_10500': 0.884881121723227, 'acc_11000': 0.8911806543385491, 'acc_11500': 0.8853891485470433, 'acc_12000': 0.8874212558423085, 'acc_12500': 0.8867100182889657, 'acc_13000': 0.8937207884576306, 'acc_13500': 0.8842714895346474, 'acc_14000': 0.8846779109937005, 'acc_14500': 0.882950619792725, 'acc_15000': 0.8964641333062385, 'acc_1500': 0.8442389758179232, 'acc_15500': 0.8905710221499695, 'acc_16000': 0.8898597845966267, 'acc_16500': 0.8910790489737859, 'acc_17000': 0.8952448689290795, 'acc_17500': 0.8974801869538712, 'acc_18000': 0.8974801869538712, 'acc_18500': 0.9001219264377159, 'acc_19000': 0.900934769355822, 'acc_19500': 0.8963625279414753, 'acc_20000': 0.9002235318024792, 'acc_2000': 0.8363137573663889, 'acc_20500': 0.8993090835196098, 'acc_21000': 0.8983946352367405, 'acc_21500': 0.8989026620605568, 'acc_22000': 0.890367811420443, 'acc_22500': 0.904490957122536, 'acc_23000': 0.9007315586262955, 'acc_23500': 0.8996138996138996, 'acc_24000': 0.8989026620605568, 'acc_24500': 0.9002235318024792, 'acc_25000': 0.9033732981101402, 'acc_2500': 0.842206868522658, 'acc_25500': 0.8973785815891079, 'acc_26000': 0.9031700873806137, 'acc_26500': 0.8936191830928673, 'acc_27000': 0.9020524283682179, 'acc_27500': 0.8977850030481609, 'acc_28000': 0.8996138996138996, 'acc_28500': 0.8961593172119487, 'acc_29000': 0.9043893517577728, 'acc_29500': 0.904490957122536, 'acc_30000': 0.8983946352367405, 'acc_3000': 0.8524690103637472, 'acc_30500': 0.9051005893111156, 'acc_31000': 0.9035765088396668, 'acc_31500': 0.9020524283682179, 'acc_32000': 0.9058118268644585, 'acc_32500': 0.9067262751473277, 'acc_33000': 0.906421459053038, 'acc_33500': 0.9024588498272709, 'acc_34000': 0.9070310912416175, 'acc_34500': 0.9077423287949604, 'acc_35000': 0.9093680146311726, 'acc_3500': 0.8647632595001016, 'acc_35500': 0.9096728307254623, 'acc_36000': 0.90835196098354, 'acc_36500': 0.9053038000406421, 'acc_37000': 0.9037797195691932, 'acc_37500': 0.9078439341597236, 'acc_38000': 0.9082503556187767, 'acc_38500': 0.9065230644178013, 'acc_39000': 0.9085551717130664, 'acc_39500': 0.9086567770778297, 'acc_40000': 0.9070310912416175, 'acc_4000': 0.8695387116439748, 'acc_40500': 0.9088599878073562, 'acc_41000': 0.9062182483235115, 'acc_41500': 0.9100792521845154, 'acc_42000': 0.9081487502540134, 'acc_42500': 0.9062182483235115, 'acc_43000': 0.9089615931721194, 'acc_43500': 0.9110953058321479, 'acc_44000': 0.9112985165616745, 'acc_44500': 0.9122129648445438, 'acc_45000': 0.9079455395244869, 'acc_4500': 0.8673033936191831, 'acc_45500': 0.9105872790083316, 'acc_46000': 0.9118065433854907, 'acc_46500': 0.9137370453159926, 'acc_47000': 0.911908148750254, 'acc_47500': 0.9114001219264377, 'acc_48000': 0.9108920951026214, 'acc_48500': 0.9129242023978866, 'acc_49000': 0.9124161755740703, 'acc_49500': 0.9133306238569396, 'acc_50000': 0.9123145702093071, 'acc_5000': 0.8603942288152815, 'acc_500': 0.8199552936395041, 'acc_50500': 0.9126193863035968, 'acc_51000': 0.9123145702093071, 'acc_51500': 0.9121113594797805, 'acc_5500': 0.8709611867506605, 'acc_6000': 0.8804104856736436, 'acc_6500': 0.8718756350335297, 'acc_7000': 0.8756350335297703, 'acc_7500': 0.8712660028449503, 'acc_8000': 0.8874212558423085, 'acc_8500': 0.8681162365372892, 'acc_9000': 0.8828490144279618, 'acc_9500': 0.8843730948994107, 'acc_model_output': 0.9121113594797805
}

python snli_multi_task.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 64 --evaluate_during_training  --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 3.0 --seed 420 --output_dir model_output --overwrite_output_dir --logging_steps 500 --save_steps 1000

python snli.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 64 --evaluate_during_training  --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 3.0 --seed 420 --output_dir model_output --overwrite_output_dir --logging_steps 500 --save_steps 1000

2020/05/20 18:57
python snli_multi_task.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 64 --evaluate_during_training  --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 3.0 --seed 76 --output_dir model_output --overwrite_output_dir --logging_steps 500 --save_steps 1000

2020/05/21 10:43
python snli.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 64 --evaluate_during_training  --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 3.0 --seed 76 --output_dir model_output --overwrite_output_dir --logging_steps 500 --save_steps 1000

2020/05/21 22:07
python snli_multi_task.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 64 --evaluate_during_training  --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 3.0 --seed 108 --output_dir model_output --overwrite_output_dir --logging_steps 500 --save_steps 1000

2020/05/22 16:41
python snli.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 64 --evaluate_during_training  --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 3.0 --seed 108 --output_dir model_output --overwrite_output_dir --logging_steps 500 --save_steps 1000

2020/05/23/ 12:29
python snli_multi_task.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 64 --evaluate_during_training  --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 3.0 --seed 7 --output_dir model_output --overwrite_output_dir --logging_steps 500 --save_steps 1000

2020/05/24 11:11
python snli.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 64 --evaluate_during_training  --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 3.0 --seed 7 --output_dir model_output --overwrite_output_dir --logging_steps 500 --save_steps 1000

2020/05/25 12:25
python snli_multi_task.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 64 --evaluate_during_training  --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 3.0 --seed 9 --output_dir model_output --overwrite_output_dir --logging_steps 500 --save_steps 1000

2020/05/26 08:29
python snli.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 64 --evaluate_during_training  --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 3.0 --seed 9 --output_dir model_output --overwrite_output_dir --logging_steps 500 --save_steps 1000


# Train data 50000
## Multi
### 2020/05/30 01:17 Seed-42
python snli_multi_task.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 64 --evaluate_during_training  --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 5.0 --seed 42 --sample_size 54000 --output_dir model_output --overwrite_output_dir --logging_steps 500 --save_steps 1000
### 2020/05/30 17:49 Seed-420
python snli_multi_task.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 64 --evaluate_during_training  --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 5.0 --seed 420 --sample_size 54000 --output_dir model_output --overwrite_output_dir --logging_steps 500 --save_steps 1000
### 2020/05/30 23:04 Seed-9
python snli_multi_task.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 64 --evaluate_during_training  --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 5.0 --seed 9 --sample_size 54000 --output_dir model_output --overwrite_output_dir --logging_steps 500 --save_steps 1000
### 2020/05/30 23:04 Seed-34
python snli_multi_task.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 64 --evaluate_during_training  --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 5.0 --seed 34 --sample_size 54000 --output_dir model_output --overwrite_output_dir --logging_steps 500 --save_steps 1000
### 2020/05/30 17:27 Seed-78
python snli_multi_task.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 64 --evaluate_during_training  --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 5.0 --seed 78 --sample_size 54000 --output_dir model_output --overwrite_output_dir --logging_steps 500 --save_steps 1000

## Single
### 05/30 11:26 Seed-42
python snli.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 64 --evaluate_during_training  --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 5.0 --seed 42 --sample_size 54000 --output_dir model_output --overwrite_output_dir --logging_steps 500 --save_steps 1000
### 05/30 20:16 Seed-420
python snli.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 64 --evaluate_during_training  --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 5.0 --seed 420 --sample_size 54000 --output_dir model_output --overwrite_output_dir --logging_steps 500 --save_steps 1000
### 05/31 01:40 Seed-9
python snli.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 64 --evaluate_during_training  --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 5.0 --seed 9 --sample_size 54000 --output_dir model_output --overwrite_output_dir --logging_steps 500 --save_steps 1000
### 05/31 14:35 Seed-34
python snli.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 64 --evaluate_during_training  --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 5.0 --seed 34 --sample_size 54000 --output_dir model_output --overwrite_output_dir --logging_steps 500 --save_steps 1000
### 05/31 21:08 Seed-78
python snli.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 64 --evaluate_during_training  --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 5.0 --seed 78 --sample_size 54000 --output_dir model_output --overwrite_output_dir --logging_steps 500 --save_steps 1000


# Train data 5000
## Multi
### 06/01 13:10 Seed-56
### 06/01 14:47 Seed-3
### 06/01 16:09 Seed-81 X
### 06/01 16:59 Seed-79 ?
### 06/01 18:38 Seed-77
python snli_multi_task.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 64 --evaluate_during_training  --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 8.0 --seed 77 --sample_size 5400 --output_dir model_output --overwrite_output_dir --logging_steps 100 --save_steps 200

### 06/01 14:20 Seed-56
### 06/01 15:23 Seed-3
### 06/01 16:46 Seed-81 X
### 06/01 17:48 Seed-79 ?
### 06/01 19:24 Seed-77
python snli.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 64  --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 8.0 --seed 77 --sample_size 5400 --output_dir model_output --overwrite_output_dir --logging_steps 100 --save_steps 200

python snli_multi_task.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 64   --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 8.0 --seed 31 --sample_size 5400 --output_dir model_output --overwrite_output_dir --logging_steps 100 --save_steps 200


# Train data 500
## Multi

### 137 68.7 
### 17 71.0
### 49 67.6
python snli_multi_task.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 64 --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 10.0 --seed 49 --sample_size 540 --output_dir model_output --overwrite_output_dir --logging_steps 33 --save_steps 330


### 137 66.1
### 17 33.8
### 49 62.8
python snli.py --model_type xlnet --model_name_or_path xlnet-base-uncased --do_train --do_eval --data_dir data/ --max_seq_length 64 --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 10.0 --seed 49 --sample_size 540 --output_dir model_output --overwrite_output_dir --logging_steps 33 --save_steps 330