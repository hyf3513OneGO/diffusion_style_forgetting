import torch
def generate_img(prompts,
                 tokenizer,
                 text_encoder,
                 scheduler,
                 unet,
                 vae,
                 device,
                 gen_shape=(512, 512),
                 num_inference_steps=4,
                 guidance_scale=0.75,
                 requires_grad=True
                 ):
    text_input = tokenizer(
        prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    batch_size = len(prompts)
    height, width = gen_shape
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // 8, width // 8),
        device=device
    )
    latents = latents * scheduler.init_noise_sigma
    from tqdm.auto import tqdm

    scheduler.set_timesteps(num_inference_steps)

    for t in scheduler.timesteps:
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # predict the noise residual
        if requires_grad:
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        else:
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    latents = 1 / 0.18215 * latents
    if requires_grad:
        image = vae.decode(latents).sample
        return image

    else:
        with torch.no_grad():
            image = vae.decode(latents).sample
            return image
