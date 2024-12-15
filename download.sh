python -m data_downloader.data_asset_download --split custom --video_id_csv test.csv --download_dir data/ --dataset_assets laser_scan_5mm annotations descriptions hires_wide hires_wide_intrinsics hires_depth hires_poses	
# python -m data_downloader.data_asset_download --split train_val_set --download_dir data/ --dataset_assets laser_scan_5mm annotations descriptions hires_wide hires_wide_intrinsics hires_depth hires_poses	
python -m data_downloader.data_asset_download --split custom --video_id_csv test.csv --download_dir data/ --dataset_assets hires_wide
