# analytic attack on ViT (April) with class attack

# one_shot_ba v1:
python classattack_breaches.py name=aprilattack_one_shot_ba_v1_b04_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=4 attack=april_analytic case.server.name=class_malicious_parameters case.model=vit_small_april case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100 base_dir=/cmlscratch/ywen/breaching/breaching/outputs save_reconstruction=True +one_shot_ba=True num_trials=50
python classattack_breaches.py name=aprilattack_one_shot_ba_v1_b08_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=8 attack=april_analytic case.server.name=class_malicious_parameters case.model=vit_small_april case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100 base_dir=/cmlscratch/ywen/breaching/breaching/outputs save_reconstruction=True +one_shot_ba=True num_trials=50
python classattack_breaches.py name=aprilattack_one_shot_ba_v1_b16_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=16 attack=april_analytic case.server.name=class_malicious_parameters case.model=vit_small_april case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100 base_dir=/cmlscratch/ywen/breaching/breaching/outputs save_reconstruction=True +one_shot_ba=True num_trials=50
python classattack_breaches.py name=aprilattack_one_shot_ba_v1_b32_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=32 attack=april_analytic case.server.name=class_malicious_parameters case.model=vit_small_april case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100 base_dir=/cmlscratch/ywen/breaching/breaching/outputs save_reconstruction=True +one_shot_ba=True num_trials=50
python classattack_breaches.py name=aprilattack_one_shot_ba_v1_b64_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=64 attack=april_analytic case.server.name=class_malicious_parameters case.model=vit_small_april case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100 base_dir=/cmlscratch/ywen/breaching/breaching/outputs save_reconstruction=True +one_shot_ba=True num_trials=50

# Unique batches:
# one_shot_ba v1:
# python classattack_breaches.py name=aprilattack_one_shot_ba_v1_b01_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=1 attack=april_analytic case.server.name=class_malicious_parameters case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100 base_dir=/cmlscratch/ywen/breaching/breaching/outputs save_reconstruction=True +one_shot_ba=True num_trials=50
python classattack_breaches.py name=aprilattack_one_shot_ba_v1_b04_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=4 attack=april_analytic case.server.name=class_malicious_parameters case.model=vit_small_april case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100 base_dir=/cmlscratch/ywen/breaching/breaching/outputs save_reconstruction=True +one_shot_ba=True num_trials=50
python classattack_breaches.py name=aprilattack_one_shot_ba_v1_b08_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=8 attack=april_analytic case.server.name=class_malicious_parameters case.model=vit_small_april case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100 base_dir=/cmlscratch/ywen/breaching/breaching/outputs save_reconstruction=True +one_shot_ba=True num_trials=50
python classattack_breaches.py name=aprilattack_one_shot_ba_v1_b16_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=16 attack=april_analytic case.server.name=class_malicious_parameters case.model=vit_small_april case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100 base_dir=/cmlscratch/ywen/breaching/breaching/outputs save_reconstruction=True +one_shot_ba=True num_trials=50
python classattack_breaches.py name=aprilattack_one_shot_ba_v1_b32_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=32 attack=april_analytic case.server.name=class_malicious_parameters case.model=vit_small_april case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100 base_dir=/cmlscratch/ywen/breaching/breaching/outputs save_reconstruction=True +one_shot_ba=True num_trials=50
python classattack_breaches.py name=aprilattack_one_shot_ba_v1_b48_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=48 attack=april_analytic case.server.name=class_malicious_parameters case.model=vit_small_april case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100 base_dir=/cmlscratch/ywen/breaching/breaching/outputs save_reconstruction=True +one_shot_ba=True num_trials=50
python classattack_breaches.py name=aprilattack_one_shot_ba_v1_b50_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=50 attack=april_analytic case.server.name=class_malicious_parameters case.model=vit_small_april case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100 base_dir=/cmlscratch/ywen/breaching/breaching/outputs save_reconstruction=True +one_shot_ba=True num_trials=50

# balanced batches:
# one_shot_ba v1:
python classattack_breaches.py name=aprilattack_clsattack_v1_b01_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=1 attack=april_analytic case.server.name=class_malicious_parameters case.model=vit_small_april case/data=ImageNet case.user.provide_labels=True case.data.partition=balanced case.data.default_clients=100 base_dir=/cmlscratch/ywen/breaching/breaching/outputs save_reconstruction=True num_trials=50
python classattack_breaches.py name=aprilattack_clsattack_v1_b32_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=32 attack=april_analytic case.server.name=class_malicious_parameters case.model=vit_small_april case/data=ImageNet case.user.provide_labels=True case.data.partition=balanced case.data.default_clients=100 base_dir=/cmlscratch/ywen/breaching/breaching/outputs save_reconstruction=True num_trials=50
python classattack_breaches.py name=aprilattack_clsattack_ba_v1_b48_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=48 attack=april_analytic case.server.name=class_malicious_parameters case.model=vit_small_april case/data=ImageNet case.user.provide_labels=True case.data.partition=balanced case.data.default_clients=100 base_dir=/cmlscratch/ywen/breaching/breaching/outputs save_reconstruction=True num_trials=50
python classattack_breaches.py name=aprilattack_clsattack_ba_v1_b50_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=50 attack=april_analytic case.server.name=class_malicious_parameters case.model=vit_small_april case/data=ImageNet case.user.provide_labels=True case.data.partition=balanced case.data.default_clients=100 base_dir=/cmlscratch/ywen/breaching/breaching/outputs save_reconstruction=True num_trials=50