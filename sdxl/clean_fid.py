from cleanfid import fid

real = r"D:\datasets\COCO\val2014_selected"
gen  = r"D:\VSC_research\generative-models\sdxl\generated_image_sdxl_512_no_sc_3_5k"

score = fid.compute_fid(
    gen, real,               
    mode="clean",
    model_name="inception_v3",
    batch_size=64,
    num_workers=0,    
    device="cuda"
)
print("Clean-FID:", score)
