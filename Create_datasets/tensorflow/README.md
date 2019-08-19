# create_csv_file.py

```bash
python create_csv_file.py --data_path ../../Datasets/my-data/train --csv_path ../../Datasets/my-data/train/train.csv
```

# create_tf_record.py

* Before run create_tf_record.py, you need to generate labels for original images, such as the path and corresponding label of each image.
you should run following shell to generate a .txt file:

```bash
python create_labels_file.py --data_path ../../Datasets/my-data/train --label_path ../../Datasets/my-data/train.txt
```

* Then run the following shell to generate tfrecords file
```bash
python create_tf_records.py --data_path ../../Datasets/my-data/train --label_path ../../Datasets/my-data/train.txt --tfrecords_path ../../Datasets/my-data/train/train.tfrecords
```
