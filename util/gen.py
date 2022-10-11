import numpy as np
from PIL import Image
import string
import random
import math


from scipy.stats import hmean

import nltk
import inspect

import torch
from torch.nn.parallel import DataParallel

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    LMSDiscreteScheduler,
    DDIMScheduler,
    PNDMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
    DiffusionPipeline
)

import clip as clip_module

from transformers import GPT2Tokenizer
from . import const, lm
lm.load_openai_key()
lm_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

MAX_LENGTH = 77

import spacy
nlp = spacy.load("en_core_web_sm")
import nltk
# nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

# These are defaults taken from huggingface/diffusers
def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def truncate_prompt(prompt, m):
    input_ids = m.tokenizer(prompt).input_ids
    if len(input_ids) >= MAX_LENGTH:
        return m.tokenizer.decode(m.tokenizer(prompt).input_ids[1:MAX_LENGTH-1])
    else:
        return m.tokenizer.decode(m.tokenizer(prompt).input_ids[1:-1])

def make_scheduler(scheduler):
    if scheduler.lower() == 'lms':
        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    elif scheduler.lower() == 'ddim':
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    else:
        scheduler = PNDMScheduler(beta_end=0.012, beta_schedule='scaled_linear', beta_start=0.00085, skip_prk_steps=True)
    return scheduler

def load(device, scheduler, gen_model="CompVis/stable-diffusion-v1-4", encoder_model="openai/clip-vit-large-patch14"):
    if gen_model == "CompVis/stable-diffusion-v1-4":
        vae = AutoencoderKL.from_pretrained(gen_model, subfolder="vae", use_auth_token=True)
        tokenizer = CLIPTokenizer.from_pretrained(encoder_model)
        text_encoder = CLIPTextModel.from_pretrained(encoder_model)
        unet = UNet2DConditionModel.from_pretrained(gen_model, subfolder="unet", use_auth_token=True)
        scheduler = make_scheduler(scheduler)
    elif gen_model == "CompVis/ldm-text2im-large-256":
        ldm = DiffusionPipeline.from_pretrained(gen_model)
        vae = ldm.vqvae
        tokenizer = ldm.tokenizer
        text_encoder = ldm.bert
        unet = ldm.unet
        scheduler = make_scheduler(scheduler)
    else:
        raise ValueError(f'model {gen_model} has not been added to available models yet')
    clip, preprocess = clip_module.load("ViT-L/14")
    
    vae = vae.to(device).eval()
    text_encoder = text_encoder.to(device).eval()
    unet = unet.to(device).eval()
    clip = clip.to(device).eval()

    return  vae, tokenizer, text_encoder, unet, scheduler, clip, preprocess

def clip_embed_text(text, m, also_unnormalized=False):
    text_features = m.clip.encode_text(clip_module.tokenize(text).to(m.device)).float()
    norm_text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    if also_unnormalized:
        return norm_text_features, text_features
    else:
        return norm_text_features

def clip_embed_image(image, m, also_unnormalized=False):
    image_features = m.clip.encode_image(m.preprocess(image).unsqueeze(0).to(m.device)).float()
    norm_image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    if also_unnormalized:
        return norm_image_features, image_features
    else:
        return norm_image_features

# clip cosine similarity
def cc(e1, e2):
    return (e1 @ e2.T).item()

def render_latents(latents, vae):
     latents = 1 / 0.18215 * latents
     image = vae.decode(latents).sample
     image = (image / 2 + 0.5).clamp(0, 1)
     image = image.cpu().permute(0, 2, 3, 1).numpy()
     image = (image * 255).round().astype("uint8")
     pil_image = [Image.fromarray(im) for im in image][0]
     return pil_image

class Cand:
    def __init__(self, score, image, seed, prompt, text_embed, image_embed, extra=None):
        self.score = score
        self.image = image
        self.seed = seed
        self.prompt = prompt
        self.text_embed = text_embed
        self.image_embed = image_embed
        self.extra = extra

def realize_prompt(prompt, seeds, steps, m, guidance_scale=7.5, size=(512, 512)):
    batch_size = 1 
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    with torch.no_grad(): 
      for seed in seeds:
        generator = torch.Generator(device=m.device).manual_seed(seed)
        m.scheduler.set_timesteps(steps)

        init_latents = torch.randn(
          (batch_size, m.unet.in_channels, height // 8, width // 8),
          generator=generator,
          device=m.device
          )
        if type(m.scheduler) is LMSDiscreteScheduler:
            init_latents = init_latents * m.scheduler.sigmas[0]

        max_length = MAX_LENGTH
        # get prompt text embeddings
        prompt_input = m.tokenizer(
            [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            text_embeddings = prompt_embeddings
        else:
            uncond_input = m.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = m.text_encoder(uncond_input.input_ids.to(m.device))[0]
            text_embeddings = torch.cat([uncond_embeddings, prompt_embeddings])

        latents = init_latents
        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]

        pil_image = render_latents(latents, m.vae)
        image_embed = clip_embed_image(pil_image, m)
        score = (text_embed @ image_embed.T).item()
        data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed))
    return sorted(data, key=lambda c: c.score, reverse=True)

def realize_prompt_batch(prompt, seeds, steps, m, guidance_scale=7.5, size=(512, 512), batch_size=1):
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    batch = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        batch.append(seed)
        if s_idx == len(seeds)-1:
            batch_size = len(batch)
        elif len(batch) < batch_size:
            continue

        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        
        latent_height, latent_width = height // 8, width // 8
        in_channels = m.unet.in_channels

        init_latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            init_latents = init_latents * m.scheduler.sigmas[0]

        max_length = MAX_LENGTH
        # get prompt text embeddings
        prompt_input = m.tokenizer(
            [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            text_embeddings = prompt_embeddings
        else:
            uncond_input = m.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = m.text_encoder(uncond_input.input_ids.to(m.device))[0]
            text_embeddings = torch.cat([uncond_embeddings, prompt_embeddings])

        latents = init_latents
        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*2) if do_classifier_free_guidance else latents
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True)

def rpr_batch(prompt, seeds, steps, m, guidance_scale=7.5, size=(512, 512), batch_size=1):
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    batch = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        batch.append(seed)
        if s_idx == len(seeds)-1:
            batch_size = len(batch)
        elif len(batch) < batch_size:
            continue

        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        
        latent_height, latent_width = height // 8, width // 8
        in_channels = m.unet.in_channels

        init_latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            init_latents = init_latents * m.scheduler.sigmas[0]

        max_length = MAX_LENGTH
        # get prompt text embeddings
        prompt_input = m.tokenizer(
            [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            text_embeddings = prompt_embeddings
        else:
            uncond_input = m.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = m.text_encoder(uncond_input.input_ids.to(m.device))[0]

            shuff_prompt = prompt.split()
            # random.Random(0).shuffle(shuff_prompt)
            shuff_prompt = shuff_prompt[::-1]
            shuff_prompt = ' '.join(shuff_prompt)
            shuff_input = m.tokenizer(
                [shuff_prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            shuff_embeddings = m.text_encoder(shuff_input.input_ids.to(m.device))[0]

            text_embeddings = torch.cat([shuff_embeddings, uncond_embeddings, prompt_embeddings])

        latents = init_latents
        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*3) if do_classifier_free_guidance else latents
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                noise_pred_shuff, noise_pred_uncond, noise_pred_text = noise_pred.chunk(3)
                noise_pred = noise_pred_uncond  + guidance_scale * 0.5 * (noise_pred_text - noise_pred_uncond)+ guidance_scale * 0.5 *(noise_pred_text - noise_pred_shuff)
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True)


def rpmg_batch(prompt, guide, seeds, steps, m, guidance_scale=7.5, size=(512, 512), batch_size=1, alpha=0.2,  mix=False):
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    batch = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        batch.append(seed)
        if s_idx == len(seeds)-1:
            batch_size = len(batch)
        elif len(batch) < batch_size:
            shuff_embeddings = m.text_encoder(torch.LongTensor([shuff_input_ids]).to(m.device))[0]
            text_embeddings = torch.cat([shuff_embeddings, uncond_embeddings, prompt_embeddings])

        latents = init_latents
        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*3) if do_classifier_free_guidance else latents
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                noise_pred_shuff, noise_pred_uncond, noise_pred_text = noise_pred.chunk(3)
                noise_pred = noise_pred_uncond  + guidance_scale * (noise_pred_text - noise_pred_uncond)+ guidance_scale * (noise_pred_text - noise_pred_shuff)
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True)


def rpmg_batch(prompt, guide, seeds, steps, m, guidance_scale=7.5, size=(512, 512), batch_size=1, alpha=0.2,  mix=False):
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    batch = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        batch.append(seed)
        if s_idx == len(seeds)-1:
            batch_size = len(batch)
        elif len(batch) < batch_size:
            continue

        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        
        latent_height, latent_width = height // 8, width // 8
        in_channels = m.unet.in_channels

        init_latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            init_latents = init_latents * m.scheduler.sigmas[0]

        max_length = MAX_LENGTH
        # get prompt text embeddings
        prompt_input = m.tokenizer(
            [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            text_embeddings = prompt_embeddings
        else:
            uncond_input = m.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = m.text_encoder(uncond_input.input_ids.to(m.device))[0]
            guide_input = m.tokenizer(
                [guide] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            guide_embeddings = m.text_encoder(guide_input.input_ids.to(m.device))[0]
            text_embeddings = torch.cat([guide_embeddings, uncond_embeddings, prompt_embeddings])

        latents = init_latents
        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*3) if do_classifier_free_guidance else latents
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                noise_pred_guide, noise_pred_uncond, noise_pred_text = noise_pred.chunk(3)
                if mix:
                    noise_pred = noise_pred_uncond + guidance_scale * (1-alpha) * (noise_pred_text - noise_pred_uncond) + guidance_scale * (alpha) * (noise_pred_guide - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) + guidance_scale * alpha * (noise_pred_guide - noise_pred_text)
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True)

def img2img(init_images, prompts, seeds, total_steps, steps, m, guidance_scale=7.5, size=(512, 512)):
    with torch.no_grad():
        num_inference_steps = total_steps
        strength = steps / total_steps

        if isinstance(prompts, str):
            batch_size = 1
            assert(not isinstance(init_images, list))
            assert(not isinstance(seeds, list))
            init_images = [init_images]
            prompts = [prompts]
            seeds = [seeds]
        elif isinstance(prompts, list):
            batch_size = len(prompts)
            assert(len(init_images) == len(prompts))
            assert(len(seeds) == len(prompts))
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        # set timesteps
        m.scheduler.set_timesteps(num_inference_steps)

        init_latents, generators = [], []
        for i, init_image in enumerate(init_images):
            if isinstance(init_image, Image.Image):
                init_image = preprocess(init_image)
            # encode the init image into latents and scale the latents
            init_latent_dist = m.vae.encode(init_image.to(m.device)).latent_dist

            generator = torch.Generator(device=m.device).manual_seed(seeds[i])
            generators.append(generator)

            init_latent = init_latent_dist.sample(generator=generator)
            init_latent = 0.18215 * init_latent
            init_latents.append(init_latent)
        single_latent_shape = init_latents[0].shape
        init_latents = torch.cat(init_latents, dim=1)

        # get the original timestep using init_timestep
        offset = m.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)
        if isinstance(m.scheduler, LMSDiscreteScheduler):
            timesteps = torch.tensor(
                [num_inference_steps - init_timestep] * batch_size, dtype=torch.long, device=m.device
            )
        else:
            timesteps = m.scheduler.timesteps[-init_timestep]
            timesteps = torch.tensor([timesteps] * batch_size, dtype=torch.long, device=m.device)

        # add noise to latents using the timesteps
        noise = [ torch.randn(single_latent_shape, generator=generator, device=m.device) ] 
        noise = torch.cat(noise, dim=0)
        init_latents = m.scheduler.add_noise(init_latents, noise, timesteps)

        prompt_embeds = [ clip_embed_text(prompt, m) for prompt in prompts ]
        # get prompt text embeddings
        text_input = m.tokenizer(
            [ prompt for prompt in prompts ],
            padding="max_length",
            max_length=m.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embed = m.text_encoder(text_input.input_ids.to(m.device))[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = m.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = m.text_encoder(uncond_input.input_ids.to(m.device))[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embed])

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(m.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        latents = init_latents

        t_start = max(num_inference_steps - init_timestep + offset, 0)
        for i, t in enumerate(m.scheduler.timesteps[t_start:]):
            t_index = t_start + i

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            # if we use LMSDiscreteScheduler, let's make sure latents are multiplied by sigmas
            if isinstance(m.scheduler, LMSDiscreteScheduler):
                sigma = m.scheduler.sigmas[t_index]
                # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(m.scheduler, LMSDiscreteScheduler):
                latents = m.scheduler.step(noise_pred, t_index, latents, **extra_step_kwargs).prev_sample
            else:
                latents = m.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # scale and decode the image latents with vae
        cands = []
        for i, latent in enumerate(latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            text_embed = prompt_embeds[i]
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            cand = Cand(score, pil_image, seeds[i], prompts[i], text_embed, image_embed)
            cands.append(cand)
        return sorted(cands, key=lambda c: c.score, reverse=True)


def img2img_old(init_image, prompt, seeds, total_steps, steps, m, guidance_scale=7.5, size=(512, 512)):
    batch_size = 1 
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    with torch.no_grad(): 
      for seed in seeds:
        generator = torch.Generator(device=m.device).manual_seed(seed)
        m.scheduler.set_timesteps(total_steps)

        if isinstance(init_image, Image.Image):
            init_image = preprocess(init_image)

        # encode the init image into latents and scale the latents
        init_latent_dist = m.vae.encode(init_image.to(m.device)).latent_dist
        init_latents = init_latent_dist.sample(generator=generator)
        init_latents = 0.18215 * init_latents

        if isinstance(m.scheduler, LMSDiscreteScheduler):
            timesteps = torch.tensor(
                [total_steps - steps] * batch_size, dtype=torch.long, device=m.device
            )
        else:
            timesteps = m.scheduler.timesteps[-steps]
            timesteps = torch.tensor([timesteps] * batch_size, dtype=torch.long, device=m.device)
        noise = torch.randn(init_latents.shape, generator=generator, device=m.device)
        latents = m.scheduler.add_noise(init_latents, noise, timesteps).to(m.device)
        if type(m.scheduler) is LMSDiscreteScheduler:
            latents = latents * m.scheduler.sigmas[0]

        max_length = MAX_LENGTH
        # get prompt text embeddings
        prompt_input = m.tokenizer(
            [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            text_embeddings = prompt_embeddings
        else:
            uncond_input = m.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = m.text_encoder(uncond_input.input_ids.to(m.device))[0]
            text_embeddings = torch.cat([uncond_embeddings, prompt_embeddings])

        t_start = max(steps, 0)
        for i, t in enumerate(m.scheduler.timesteps[t_start:]):
             t_index = t_start + i

             latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
             if type(m.scheduler) is LMSDiscreteScheduler:
                 sigma = m.scheduler.sigmas[t_index]
                 latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
             noise_pred = m.unet(latent_model_input.float(), t, encoder_hidden_states=text_embeddings)["sample"]
             if do_classifier_free_guidance:
                 noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                 noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

             if type(m.scheduler) is LMSDiscreteScheduler:
                 latents = m.scheduler.step(noise_pred, t_index, latents)["prev_sample"]
             else:
                 latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]

        for i, latent in enumerate(latents):
            pil_image = render_latents(latent.float(), m.vae)
            text_embed = prompt_embeds[i]
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed))
    return sorted(data, key=lambda c: c.score, reverse=True)

def add_prefix(prompt):
    return f'a photo of {prompt.strip}'

def span_embeddings(s, m):
  idxs = m.tokenizer.encode(s)
  tokens = m.tokenizer.convert_ids_to_tokens(idxs)
  spans = {}
  for i in range(1, len(idxs)-1):
    for j in range(i+1, len(idxs)):
      if tokens[i-1] != '<|startoftext|>' and not tokens[i-1].endswith('</w>'):
          continue
      elif not (j == len(idxs)-1 or tokens[j-1].endswith('</w>')):
          continue
      span = tuple(idxs[i:j])
      #span_complement = tuple(idxs[:i] + idxs[j:])
      # replace span with '...'
      # span_complement = tuple(idxs[:i] + [678] + idxs[j:])
      # don't replace span wiht anything
      span_complement = tuple(idxs[:i] + idxs[j:])
      # replace span with 'something'
      # span_complement = tuple(idxs[:i] + [2006] + idxs[j:])
      if span_complement in spans:
        continue
      pad_span = [idxs[0]] + list(span) + [idxs[-1]] + [0] * (77-len(span)-2) 
      pad_span_complement = list(span_complement) + [0] * (77-len(span_complement)) 
      span_tensor = torch.tensor(pad_span, dtype=torch.int32).unsqueeze(0).cuda(m.device)
      span_complement_tensor = torch.tensor(pad_span_complement, dtype=torch.int32).unsqueeze(0).cuda(m.device)
      features = m.clip.encode_text(span_tensor).float()
      complement_features = m.clip.encode_text(span_complement_tensor).float()
      norm_features = features / features.norm(dim=-1, keepdim=True)
      norm_complement_features = complement_features / complement_features.norm(dim=-1, keepdim=True)
      span_data = { 
                      'idxs' : span,
                      'i' : i,
                      'j' : j,
                      'string' : m.tokenizer.decode(span), 
                      'features' : features,
                      'norm_features' : norm_features,
                      'span_tensor' : span_tensor,
                      'c-idxs' : span_complement,
                      'c-string' : m.tokenizer.decode(span_complement), 
                      'c-features' : complement_features,
                      'c-norm_features' : norm_complement_features,
                      'c-span_tensor' : span_complement_tensor
                }
      spans[span] = span_data
  return spans

def ignorable_spans(prompt, norm_text_features, norm_image_features, span_embeds, m, require_noun=True, epsilon=0):
    ignorable = []
    all_spans = []
    for span in span_embeds.values():
        skip = True
        for word in span['string'].split():
            if word not in stopwords and word not in string.punctuation:
                skip = False
                break
        if skip:
            continue
        missing_info_score = (span['c-norm_features'] @ norm_image_features.T).item() - (norm_text_features @ norm_image_features.T).item()
        span['score'] = missing_info_score
        if span['score'] >  -epsilon:
            ignorable.append(span)
        all_spans.append(span)
    igs = []
    if not require_noun:
        igs = ignorable
    else:
        for span in ignorable:
            if any([ token.pos_ == 'NOUN' for token in nlp(span['string']) ]):
                igs.append(span)
    #return sorted(igs, key=lambda c: (-c['score']))
    return sorted(igs, key=lambda c: -c['score']), sorted(all_spans, key=lambda c: -c['score']), 

def chunk_embeddings(s, m):
     doc = nlp(s)
     chunks = list(doc.noun_chunks)
     chunk_data = []
     for chunk in chunks:
         txt_embed = clip_embed_text(str(chunk), m)
         complement = s[0:chunk.start_char].strip() + ' ' + s[chunk.end_char:].strip()
         cmp_embed = clip_embed_text(complement, m)
         span_data = { 
                      'idxs' : m.tokenizer(str(chunk)),
                      'string' : str(chunk),
                      'norm_features' : txt_embed,
                      'c-string' : complement,
                      'c-norm_features' : cmp_embed,
                }
         chunk_data.append(span_data)
     return chunk_data

def ignorable_chunks(prompt, norm_text_features, norm_image_features, m):
    ignorable = []
    for span in chunk_embeddings(prompt, m):
        skip = True
        for word in span['string'].split():
            if word not in stopwords:
                skip = False
                break
        if skip:
            continue
        missing_info_score = (span['c-norm_features'] @ norm_image_features.T).item() - (norm_text_features @ norm_image_features.T).item()
        span['score'] = missing_info_score
        if span['score'] > 0:
            ignorable.append(span)
    return sorted(ignorable, key=lambda c: (-len(c['idxs']), -c['score']))

def chunk_probes(s):
     doc = nlp(s)
     chunks = list(doc.noun_chunks)
     probes = []
     for chunk in chunks:
         probe = s[0:chunk.start_char].strip() + ' something ' + s[chunk.end_char:].strip()
         probes.append(probe)
     return probes

def noun_chunks(s):
     doc = nlp(s)
     chunks = list(doc.noun_chunks)
     return [ str(chunk) for chunk in chunks ]


def r_iter_add(prompt, n, steps, m, guidance_scale=7.5, size=(512, 512)):
    with torch.no_grad():
        data = []
        cand = realize_prompt(prompt, [0], steps, m, guidance_scale, size)[0]
        prompt_embed = cand.text_embed
        data.append(cand)
        for i in range(1, n):
            li_span = longest_ignorable_span(prompt, cand.text_embed, cand.image_embed, m)
            if li_span is None:
                new_prompt = cand.prompt
            elif cand.prompt == prompt:
                new_prompt = f'{prompt} | keywords: {li_span["string"]}'
            else:
                new_prompt = f'{new_prompt}, {li_span["string"]}'
            cand = realize_prompt(new_prompt, [i], steps, m, guidance_scale, size)[0]
            cand.score = cc(cand.image_embed, prompt_embed)
            data.append(cand)
        return sorted(data, key=lambda c: c.score, reverse=True)

def r_two_stage_ia(prompt, k, n, init_steps, final_steps, m, guidance_scale=7.5, size=(512, 512)):
    with torch.no_grad():
        init_cands = r_iter_add(prompt, k, init_steps, m, guidance_scale=guidance_scale, size=size)
        final_inits = [ (cand.prompt, cand.seed) for cand in init_cands[:n] ]
        final_cands = [ realize_prompt(prompt, [seed], final_steps, m, guidance_scale=guidance_scale, size=size)[0] for prompt, seed in final_inits ]
        prompt_embed = clip_embed_text(prompt, m)
        for cand in final_cands:
            cand.score = cc(cand.image_embed, prompt_embed)
        return sorted(final_cands, key=lambda c: c.score, reverse=True)

def r_two_stage(prompt, k, n, init_steps, final_steps, m, guidance_scale=7.5, size=(512, 512)):
    with torch.no_grad():
        init_cands = realize_prompt(prompt, range(k), init_steps, m, guidance_scale=guidance_scale, size=size)
        final_seeds = [ cand.seed for cand in init_cands[:n] ]
        final_cands = realize_prompt(prompt, final_seeds, final_steps, m, guidance_scale=guidance_scale, size=size)
        return final_cands

def r_ts_wlm(prompt, k, n, init_steps, final_steps, m, guidance_scale=7.5, size=(512, 512)):
    with torch.no_grad():
        init_cands = [ realize_prompt(prompt, [0], init_steps, m, guidance_scale=guidance_scale, size=size)[0] ]
        for i in range(1, k):
            new_prompt = lm.gen('text-davinci-002', lm_tokenizer, f'Paraphrase and expand the following into an entire paragraph: {prompt}', 100, echo=False)[0]['text'].strip()
            new_prompt = truncate_prompt(new_prompt, m)
            new_cand = realize_prompt(new_prompt, [i], init_steps, m, guidance_scale=guidance_scale, size=size)[0]
            init_cands.append(new_cand)
        final_inits = [ (cand.prompt, cand.seed) for cand in init_cands[:n] ]
        final_cands = [ realize_prompt(prompt, [seed], final_steps, m, guidance_scale=guidance_scale, size=size)[0] for prompt, seed in final_inits ]
        prompt_embed = clip_embed_text(prompt, m)
        for cand in final_cands:
            cand.score = cc(cand.image_embed, prompt_embed)
        return sorted(final_cands, key=lambda c: c.score, reverse=True)

def r_beam_search(prompt, max_k, n, steps, rounds, m, guidance_scale=7.5, size=(512, 512)):
    cur_beam = realize_prompt(prompt, range(n), steps, m)
    prompt_embed = cur_beam[0].text_embed
    next_seed = n
    final_cands = [ cand for cand in cur_beam ]
    for r in range(rounds-1):
        next_beam = []
        for cand in cur_beam:
            for span in ignorable_spans(prompt, prompt_embed, cand.image_embed, m)[:max_k]:
                if r == 0:
                    new_prompt = f'{prompt}\nkeywords: {span["string"].strip()}'
                else:
                    new_prompt = f'{cand.prompt}, {span["string"].strip()}'
                full_prompt = len(m.tokenizer(new_prompt).input_ids) >= MAX_LENGTH
                if full_prompt:
                    new_prompt = m.tokenizer.decode(m.tokenizer(new_prompt).input_ids[1:MAX_LENGTH])
                new_cand = realize_prompt(new_prompt, [next_seed], steps, m)[0]
                new_cand.score = cc(new_cand.image_embed, prompt_embed)
                next_seed += 1
                # only consider for the next round if there is space left to expand the prompt
                if not full_prompt:
                    next_beam.append(new_cand)
                final_cands.append(new_cand)
        cur_beam = sorted(next_beam, key=lambda c: c.score, reverse=True)[:n]
    return sorted(final_cands, key=lambda c: c.score, reverse=True)

def rpg(prompt, seeds, steps, m, guidance_scale=7.5, size=(512, 512), batch_size=1, prefix='', chunks=True):
    assert(batch_size == 1)
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    with torch.no_grad(): 
      for seed in seeds:
        generator = torch.Generator(device=m.device).manual_seed(seed)
        m.scheduler.set_timesteps(steps)

        init_latents = torch.randn(
          (batch_size, m.unet.in_channels, height // 8, width // 8),
          generator=generator,
          device=m.device
          )
        if type(m.scheduler) is LMSDiscreteScheduler:
            init_latents = init_latents * m.scheduler.sigmas[0]

        max_length = MAX_LENGTH

        prompt_input = m.tokenizer(
            [f'{prefix}{prompt}'] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]

        span_embeds = span_embeddings(prompt, m)
        chunk_embeds = chunk_embeddings(prompt, m)
        # random.shuffle(chunk_embeds)
        # chunk_prompt = f'{prefix}in this image: {", ".join([ chunk["string"] for chunk in chunk_embeds ])} | {prompt}'
        # chunk_prompt = f'{prefix}{prompt} | in this image: {", ".join([ chunk["string"] for chunk in chunk_embeds ])}'
        # chunk_prompt = f'{prefix}in this image: {random.choice(chunk_embeds)["string"]} | {prompt}'
        chunk_embeds = chunk_embeds[::-1]
        chunk_idx = 0
        chunk_prompt = f'{prefix}in this image: {chunk_embeds[chunk_idx]["string"]} | {prompt}'
        chunk_idx = (chunk_idx + 1) % len(chunk_embeds)
        chunk_prompt_input = m.tokenizer(
            [chunk_prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        chunk_prompt_embeddings = m.text_encoder(chunk_prompt_input.input_ids.to(m.device))[0]

        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            text_embeddings = prompt_embeddings
        else:
            uncond_input = m.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = m.text_encoder(uncond_input.input_ids.to(m.device))[0]
            if chunks:
               text_embeddings = torch.cat([uncond_embeddings, chunk_prompt_embeddings])
            else:
                text_embeddings = torch.cat([uncond_embeddings, prompt_embeddings])


        cur_prompt_embeddings = text_embeddings
        latents = init_latents
        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]

            cur_image = render_latents(latents, m.vae)
            cur_image_embed = clip_embed_image(cur_image, m) 
            spans = ignorable_spans(prompt, text_embed, cur_image_embed, span_embeds, m)
            if len(spans) > 0 and len(spans[0]['string']) <= len(prompt) * (9/10):
                cur_prompt =  f'{prefix}in this image: {spans[0]["string"]} | {prompt}'
                # cur_prompt = f'{spans[0]["string"]}'
                cur_prompt_input = m.tokenizer(
                    [cur_prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                )
                cur_prompt_embeddings = m.text_encoder(cur_prompt_input.input_ids.to(m.device))[0]
            else:
                if chunks:
                    # random.shuffle(chunk_embeds)
                    # chunk_prompt = f'{prefix}in this image: {", ".join([ chunk["string"] for chunk in chunk_embeds ])} | {prompt}'
                    # chunk_prompt = f'{prefix}{prompt} | in this image: {", ".join([ chunk["string"] for chunk in chunk_embeds ])}'
                    # chunk_prompt = f'{prefix}in this image: {random.choice(chunk_embeds)["string"]} | {prompt}'
                    chunk_prompt = f'{prefix}in this image: {chunk_embeds[chunk_idx]["string"]} | {prompt}'
                    chunk_idx = (chunk_idx + 1) % len(chunk_embeds)
                    chunk_prompt_input = m.tokenizer(
                        [chunk_prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                    )
                    chunk_prompt_embeddings = m.text_encoder(chunk_prompt_input.input_ids.to(m.device))[0]
                    cur_prompt = chunk_prompt
                    cur_prompt_embeddings = chunk_prompt_embeddings
                else:
                    cur_prompt = prompt
                    cur_prompt_embeddings = prompt_embeddings
            text_embeddings = torch.cat([uncond_embeddings, cur_prompt_embeddings])

        pil_image = render_latents(latents, m.vae)
        image_embed = clip_embed_image(pil_image, m)
        score = (text_embed @ image_embed.T).item()
        data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed))
    return sorted(data, key=lambda c: c.score, reverse=True)


def realize_prompt_post(prompt, seeds, steps, m, guidance_scale=7.5, size=(512, 512), batch_size=1):
    seeds = list(seeds)
    extra_seed = max(seeds) + 1
    cands_plus_one = realize_prompt_batch(prompt, seeds + [extra_seed], steps, m, guidance_scale=guidance_scale, size=size, batch_size=batch_size)
    baseline_cand = cands_plus_one[0]
    cands_plus_one = sorted(cands_plus_one, key=lambda c: c.seed)
    cands = cands_plus_one[:-1]
    extra_cand = cands_plus_one[-1]
    best_cand = cands[0]
    span_embeds = span_embeddings(prompt, m)
    spans = ignorable_spans(best_cand.prompt, best_cand.text_embed, best_cand.image_embed, span_embeds, m)
    if len(spans) > 0 and len(spans[0]['string']) <= len(prompt) * (9/10):
        updated_prompt = True
        new_prompt =  f'{prompt}; visible in this image: {spans[0]["string"]}'
        hypo_cand = realize_prompt_batch(new_prompt, [best_cand.seed], steps, m, guidance_scale=guidance_scale, size=size, batch_size=batch_size)[0]
        hypo_cand.score = cc(hypo_cand.image_embed, best_cand.text_embed)
    else:
        hypo_cand = baseline_cand
    return hypo_cand, baseline_cand, cands_plus_one

def max_rev_batch(prompt, seeds, steps, m, guidance_scale=7.5, size=(512, 512), batch_size=1):
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    batch = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        batch.append(seed)
        if s_idx == len(seeds)-1:
            batch_size = len(batch)
        elif len(batch) < batch_size:
            continue

        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        
        latent_height, latent_width = height // 8, width // 8
        in_channels = m.unet.in_channels

        init_latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            init_latents = init_latents * m.scheduler.sigmas[0]

        max_length = MAX_LENGTH
        # get prompt text embeddings
        prompt_input = m.tokenizer(
            [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            text_embeddings = prompt_embeddings
        else:
            uncond_input = m.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = m.text_encoder(uncond_input.input_ids.to(m.device))[0]

            shuff_prompt = prompt.split()
            # random.Random(0).shuffle(shuff_prompt)
            shuff_prompt = shuff_prompt[::-1]
            shuff_prompt = ' '.join(shuff_prompt)
            shuff_input = m.tokenizer(
                [shuff_prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            shuff_embeddings = m.text_encoder(shuff_input.input_ids.to(m.device))[0]

            text_embeddings = torch.cat([shuff_embeddings, uncond_embeddings, prompt_embeddings])

        latents = init_latents
        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*3) if do_classifier_free_guidance else latents
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                noise_pred_shuff, noise_pred_uncond, noise_pred_text = noise_pred.chunk(3)
                C = noise_pred_text - noise_pred_uncond
                M = noise_pred_text - noise_pred_shuff
                CM = torch.cat([C, M], dim=0)
                indices = CM.abs().max(dim=0).indices
                noise_pred_ling = CM.gather(index=indices.unsqueeze(0), dim=0)
                noise_pred = noise_pred_uncond  + guidance_scale * noise_pred_ling
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True)

def shufneg_batch(prompt, seeds, steps, m, guidance_scale=7.5, size=(512, 512), batch_size=1, guide=None, weight=[0.5, 0.5]):
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    batch = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        batch.append(seed)
        if s_idx == len(seeds)-1:
            batch_size = len(batch)
        elif len(batch) < batch_size:
            continue

        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        
        latent_height, latent_width = height // 8, width // 8
        in_channels = m.unet.in_channels

        init_latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            init_latents = init_latents * m.scheduler.sigmas[0]

        max_length = MAX_LENGTH
        # get prompt text embeddings
        prompt_input = m.tokenizer(
            [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            text_embeddings = prompt_embeddings
        else:
            uncond_input = m.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = m.text_encoder(uncond_input.input_ids.to(m.device))[0]

            shuff_prompt = prompt.split()
            random.Random(0).shuffle(shuff_prompt)

            if guide is None:
                shuff_prompt = ' '.join(shuff_prompt)
                shuff_input = m.tokenizer(
                    [shuff_prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                )
                shuff_embeddings = m.text_encoder(shuff_input.input_ids.to(m.device))[0]
                text_embeddings = torch.cat([shuff_embeddings, uncond_embeddings, prompt_embeddings])
            else:
                shuff_input = m.tokenizer(
                    [guide] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                )
                shuff_embeddings = m.text_encoder(shuff_input.input_ids.to(m.device))[0]
                text_embeddings = torch.cat([shuff_embeddings, uncond_embeddings, prompt_embeddings])

        latents = init_latents
        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*3) if do_classifier_free_guidance else latents
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                noise_pred_shuff, noise_pred_uncond, noise_pred_text = noise_pred.chunk(3)
                C = noise_pred_text - noise_pred_uncond
                M = noise_pred_text - noise_pred_shuff
                CM = torch.cat([C, M], dim=0)
                weight_t = torch.Tensor(weight).view(2, 1, 1, 1).to(m.device)
                noise_pred_ling = CM.mul(weight_t).sum(0, keepdim=True)
                noise_pred = noise_pred_uncond  + guidance_scale * noise_pred_ling
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True)

def hm_seed_sort(prompt, probes, seeds, steps, m, guidance_scale=7.5, size=(512, 512)):
    height, width = size
    batch_size = 1 

    text_embed = clip_embed_text(prompt, m)
    latent_height, latent_width = height // 8, width // 8
    in_channels = m.unet.in_channels
    max_length = MAX_LENGTH
    # get prompt text embeddings
    prompt_input = m.tokenizer(
        [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
    uncond_input = m.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = m.text_encoder(uncond_input.input_ids.to(m.device))[0]
    all_probe_embeddings = []
    print(probes)
    for probe in probes:
        probe_input = m.tokenizer(
            [probe] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        probe_embeddings = m.text_encoder(probe_input.input_ids.to(m.device))[0]
        all_probe_embeddings.append(probe_embeddings)
    n = 2 + len(probes)
    text_embeddings = torch.cat([uncond_embeddings, prompt_embeddings] + all_probe_embeddings)


    batch = []
    data = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        batch.append(seed)
        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            latents = latents * m.scheduler.sigmas[0]

        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*n) 
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            noise_preds = noise_pred.chunk(n)
            U, C = noise_preds[:2]
            Ps = noise_preds[2:]
            noise_pred = U + guidance_scale*(C-U)
            probe_scores = [ (C-P_i).abs().max().item() for P_i in Ps ]
            print(probe_scores)
                
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed in batch:
            score = hmean(probe_scores)
            data.append((score, seed))
        batch = []
        print('-'*20)
    return sorted(data, reverse=True)

def hmp_seed_sort(prompt, probes, seeds, steps, m, guidance_scale=7.5, size=(512, 512)):
    height, width = size
    batch_size = 1 

    text_embed = clip_embed_text(prompt, m)
    latent_height, latent_width = height // 8, width // 8
    in_channels = m.unet.in_channels
    max_length = MAX_LENGTH
    # get prompt text embeddings
    prompt_input = m.tokenizer(
        [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
    uncond_input = m.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = m.text_encoder(uncond_input.input_ids.to(m.device))[0]
    all_probe_embeddings = []
    print(probes)
    for probe in probes:
        probe_input = m.tokenizer(
            [probe] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        probe_embeddings = m.text_encoder(probe_input.input_ids.to(m.device))[0]
        all_probe_embeddings.append(probe_embeddings)
    n = 2 + len(probes)
    text_embeddings = torch.cat([uncond_embeddings, prompt_embeddings] + all_probe_embeddings)


    batch = []
    data = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        batch.append(seed)
        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            latents = latents * m.scheduler.sigmas[0]

        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*n) 
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            noise_preds = noise_pred.chunk(n)
            U, C = noise_preds[:2]
            Ps = noise_preds[2:]
            noise_pred = U + guidance_scale*(C-U)
            probe_scores = [ (C-P_i).abs().mean().item() for P_i in Ps ]
            print(probe_scores)
                
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed in batch:
            score = hmean(probe_scores)
            data.append((score, seed))
        batch = []
        print('-'*20)
    return sorted(data)

def parastab_seed_sort(prompts, seeds, steps, m, guidance_scale=7.5, size=(512, 512)):
    height, width = size
    batch_size = 1 

    text_embed = clip_embed_text(prompts[0], m)
    latent_height, latent_width = height // 8, width // 8
    in_channels = m.unet.in_channels
    max_length = MAX_LENGTH
    uncond_input = m.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = m.text_encoder(uncond_input.input_ids.to(m.device))[0]
    prompt_embeddings = []
    for prompt in prompts:
        prompt_input = m.tokenizer(
            [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embedding_instance = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
        prompt_embeddings.append(prompt_embedding_instance)
    n = 1 + len(prompts)
    text_embeddings = torch.cat([uncond_embeddings] + prompt_embeddings)

    batch = []
    data = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        batch.append(seed)
        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            latents = latents * m.scheduler.sigmas[0]

        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*n) 
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            noise_preds = noise_pred.chunk(n)
            U = noise_preds[0]
            Cs = noise_preds[1:]
            noise_pred = U + guidance_scale*torch.cat([C-U for C in Cs], dim=0).mean(0)
                
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompts, text_embed, image_embed))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True)

def shuffstab(prompt, seeds, steps, n_shuff,  m, guidance_scale=7.5, size=(512, 512)):
    height, width = size
    batch_size = 1 

    text_embed = clip_embed_text(prompt, m)
    latent_height, latent_width = height // 8, width // 8
    in_channels = m.unet.in_channels
    max_length = MAX_LENGTH
    uncond_input = m.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embedding = m.text_encoder(uncond_input.input_ids.to(m.device))[0]
    prompt_input = m.tokenizer(
        [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    prompt_embedding = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
    stabilizers = []
    words = prompt.split()
    for stab_seed in range(n_shuff):
        stab = ' '.join(random.Random(stab_seed).sample(words, k=len(words) // 3))
        stab_input = m.tokenizer(
            [stab] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        stab_embedding = m.text_encoder(stab_input.input_ids.to(m.device))[0]
        stabilizers.append(stab_embedding)
    stab_embedding = torch.cat(stabilizers, 0).mean(0, keepdim=True)
    text_embeddings = torch.cat([uncond_embedding, prompt_embedding, stab_embedding])

    batch = []
    data = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        batch.append(seed)
        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            latents = latents * m.scheduler.sigmas[0]

        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*3) 
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            U, C, S = noise_pred.chunk(3)
            noise_pred = C + guidance_scale*(C-S)
                
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True)

def compguide(prompt, guide, seeds, steps,  m, guidance_scale=7.5, size=(512, 512)):
    height, width = size
    batch_size = 1 

    text_embed = clip_embed_text(prompt, m)
    latent_height, latent_width = height // 8, width // 8
    in_channels = m.unet.in_channels
    max_length = MAX_LENGTH
    uncond_input = m.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embedding = m.text_encoder(uncond_input.input_ids.to(m.device))[0]
    prompt_input = m.tokenizer(
        [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    prompt_embedding = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
    guide_input = m.tokenizer(
        [guide] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    guide_embedding = m.text_encoder(guide_input.input_ids.to(m.device))[0]
    text_embeddings = torch.cat([uncond_embedding, prompt_embedding, guide_embedding])

    batch = []
    data = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        batch.append(seed)
        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            latents = latents * m.scheduler.sigmas[0]

        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*3) 
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            U, C, G = noise_pred.chunk(3)
            V = torch.cat([C-U, C-G], dim=0).sum(0, keepdim=True)
            noise_pred = U + guidance_scale*V
                
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True)

def pn_guide(prompt, pos, neg, seeds, steps,  m, guidance_scale=7.5, size=(512, 512)):
    height, width = size
    batch_size = 1 

    text_embed = clip_embed_text(prompt, m)
    latent_height, latent_width = height // 8, width // 8
    in_channels = m.unet.in_channels
    max_length = MAX_LENGTH
    uncond_input = m.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embedding = m.text_encoder(uncond_input.input_ids.to(m.device))[0]
    prompt_input = m.tokenizer(
        [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    prompt_embedding = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
    pos_input = m.tokenizer(
        [pos] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    pos_embedding = m.text_encoder(pos_input.input_ids.to(m.device))[0]
    neg_input = m.tokenizer(
        [neg] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    neg_embedding = m.text_encoder(neg_input.input_ids.to(m.device))[0]
    text_embeddings = torch.cat([uncond_embedding, prompt_embedding, pos_embedding, neg_embedding])

    batch = []
    data = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        batch.append(seed)
        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            latents = latents * m.scheduler.sigmas[0]

        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*4)
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            U, C, P, N = noise_pred.chunk(4)
            G = torch.cat([P-C, C-N], dim=0).mean(0, keepdim=True)
            V = torch.cat([C-U, G], dim=0).sum(0, keepdim=True)
            noise_pred = U + guidance_scale*V
                
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True)

def cossim(t1, t2):
    vt1, vt2 = t1.view(t1.size(0), -1), t2.view(t2.size(0), -1)
    n_vt1, n_vt2  = vt1 / vt1.norm(dim=-1, keepdim=True), vt2 / vt2.norm(dim=-1, keepdim=True)
    result =  n_vt1 @ n_vt2.T
    return result

def rps_batch(prompt, seeds, steps, m, guidance_scale=7.5, size=(512, 512), batch_size=1):
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    batch = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        stats = []
        batch.append(seed)
        if s_idx == len(seeds)-1:
            batch_size = len(batch)
        elif len(batch) < batch_size:
            continue

        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        
        latent_height, latent_width = height // 8, width // 8
        in_channels = m.unet.in_channels

        init_latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            init_latents = init_latents * m.scheduler.sigmas[0]

        max_length = MAX_LENGTH
        # get prompt text embeddings
        prompt_input = m.tokenizer(
            [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            text_embeddings = prompt_embeddings
        else:
            uncond_input = m.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = m.text_encoder(uncond_input.input_ids.to(m.device))[0]
            text_embeddings = torch.cat([uncond_embeddings, prompt_embeddings])

        latents = init_latents
        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*2) if do_classifier_free_guidance else latents
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                U, C = noise_pred.chunk(2)
                stats.append((U-C).abs().sum((1, 2, 3)))
                s = guidance_scale
                noise_pred = U + s*(C-U)
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed, extra=cossim(U, C)))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True), stats

def counter_batch(prompt, seeds, steps, m, guidance_scale=7.5, size=(512, 512), batch_size=1):
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    batch = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        stats = []
        batch.append(seed)
        if s_idx == len(seeds)-1:
            batch_size = len(batch)
        elif len(batch) < batch_size:
            continue

        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        
        latent_height, latent_width = height // 8, width // 8
        in_channels = m.unet.in_channels

        init_latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            init_latents = init_latents * m.scheduler.sigmas[0]

        max_length = MAX_LENGTH
        # get prompt text embeddings
        prompt_input = m.tokenizer(
            [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            text_embeddings = prompt_embeddings
        else:
            words = prompt.split()
            counter_embeddings = []
            for i in range(len(words)):
                counter_words = words[i]
                counter_input = m.tokenizer(
                    [counter_words] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                )
                counter_embeddings.append(m.text_encoder(counter_input.input_ids.to(m.device))[0])
            counter_embeddings = torch.stack(counter_embeddings, dim=0).mean(dim=0)
            text_embeddings = torch.cat([counter_embeddings, prompt_embeddings])

        latents = init_latents
        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*2) if do_classifier_free_guidance else latents
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                U, C = noise_pred.chunk(2)
                stats.append((U-C).abs().sum((1, 2, 3)))
                s = guidance_scale
                noise_pred = U + s*(C-U)
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed, extra=cossim(U, C)))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True), stats


def neg_batch(prompt, neg, seeds, steps, m, guidance_scale=7.5, size=(512, 512), batch_size=1):
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    batch = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        stats = []
        batch.append(seed)
        if s_idx == len(seeds)-1:
            batch_size = len(batch)
        elif len(batch) < batch_size:
            continue

        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        
        latent_height, latent_width = height // 8, width // 8
        in_channels = m.unet.in_channels

        init_latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            init_latents = init_latents * m.scheduler.sigmas[0]

        max_length = MAX_LENGTH
        # get prompt text embeddings
        prompt_input = m.tokenizer(
            [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            text_embeddings = prompt_embeddings
        else:
            counter_input = m.tokenizer(
                [neg] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            counter_embeddings = m.text_encoder(counter_input.input_ids.to(m.device))[0]
            text_embeddings = torch.cat([counter_embeddings, prompt_embeddings])

        latents = init_latents
        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*2) if do_classifier_free_guidance else latents
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                N, C = noise_pred.chunk(2)
                s = guidance_scale
                noise_pred = C + s*(C-N)
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed, extra=cossim(N, C)))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True), stats

def pos_batch(prompt, pos, seeds, steps, m, guidance_scale=7.5, size=(512, 512), batch_size=1):
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    batch = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        stats = []
        batch.append(seed)
        if s_idx == len(seeds)-1:
            batch_size = len(batch)
        elif len(batch) < batch_size:
            continue

        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        
        latent_height, latent_width = height // 8, width // 8
        in_channels = m.unet.in_channels

        init_latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            init_latents = init_latents * m.scheduler.sigmas[0]

        max_length = MAX_LENGTH
        # get prompt text embeddings
        prompt_input = m.tokenizer(
            [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            text_embeddings = prompt_embeddings
        else:
            counter_input = m.tokenizer(
                [pos] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            counter_embeddings = m.text_encoder(counter_input.input_ids.to(m.device))[0]
            text_embeddings = torch.cat([counter_embeddings, prompt_embeddings])

        latents = init_latents
        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*2) if do_classifier_free_guidance else latents
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                P, C = noise_pred.chunk(2)
                s = guidance_scale
                noise_pred = C + s*(P-C)
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed, extra=cossim(P, C)))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True), stats


def rev_batch(prompt, seeds, steps, m, guidance_scale=7.5, size=(512, 512), batch_size=1):
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    batch = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        stats = []
        batch.append(seed)
        if s_idx == len(seeds)-1:
            batch_size = len(batch)
        elif len(batch) < batch_size:
            continue

        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        
        latent_height, latent_width = height // 8, width // 8
        in_channels = m.unet.in_channels

        init_latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            init_latents = init_latents * m.scheduler.sigmas[0]

        max_length = MAX_LENGTH
        # get prompt text embeddings
        prompt_input = m.tokenizer(
            [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            text_embeddings = prompt_embeddings
        else:
            counter = ' '.join(prompt.split()[::-1])
            counter_input = m.tokenizer(
                [counter] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            counter_embeddings = m.text_encoder(counter_input.input_ids.to(m.device))[0]
            text_embeddings = torch.cat([counter_embeddings, prompt_embeddings])

        latents = init_latents
        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*2) if do_classifier_free_guidance else latents
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                P, C = noise_pred.chunk(2)
                s = guidance_scale
                noise_pred = P + s*(C-P)
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed, extra=cossim(P, C)))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True), stats

def bits_batch(prompt, seeds, steps, m, guidance_scale=7.5, size=(512, 512), batch_size=1):
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    batch = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        stats = []
        batch.append(seed)
        if s_idx == len(seeds)-1:
            batch_size = len(batch)
        elif len(batch) < batch_size:
            continue

        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        
        latent_height, latent_width = height // 8, width // 8
        in_channels = m.unet.in_channels

        init_latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            init_latents = init_latents * m.scheduler.sigmas[0]

        max_length = MAX_LENGTH
        # get prompt text embeddings
        prompt_input = m.tokenizer(
            [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            text_embeddings = prompt_embeddings
        else:
            words = prompt.split()
            counter_embeddings = []
            l = 3
            for i in range(len(words)-l):
                counter_words = ' '.join(words[i:i+l])
                counter_input = m.tokenizer(
                    [counter_words] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                )
                counter_embeddings.append(m.text_encoder(counter_input.input_ids.to(m.device))[0])
            counter_embeddings = torch.stack(counter_embeddings, dim=0).mean(dim=0)
            text_embeddings = torch.cat([counter_embeddings, prompt_embeddings])

        latents = init_latents
        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*2) if do_classifier_free_guidance else latents
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                U, C = noise_pred.chunk(2)
                stats.append((U-C).abs().sum((1, 2, 3)))
                s = guidance_scale
                noise_pred = U + s*(C-U)
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed, extra=cossim(U, C)))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True), stats

def clip_batch(prompt, seeds, steps, m, guidance_scale=7.5, size=(512, 512), batch_size=1):
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    batch = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        stats = []
        batch.append(seed)
        if s_idx == len(seeds)-1:
            batch_size = len(batch)
        elif len(batch) < batch_size:
            continue

        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        
        latent_height, latent_width = height // 8, width // 8
        in_channels = m.unet.in_channels

        init_latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            init_latents = init_latents * m.scheduler.sigmas[0]

        max_length = MAX_LENGTH
        # get prompt text embeddings
        prompt_input = m.tokenizer(
            [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
        text_embed = clip_embed_text(prompt, m)
        span_embeds = span_embeddings(prompt, m)
        n = len(m.tokenizer(prompt)['input_ids'])
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            text_embeddings = prompt_embeddings
        else:
            uncond_input = m.tokenizer(
                [""], padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embedding = m.text_encoder(uncond_input.input_ids.to(m.device))[0]

        latents = init_latents
        for i, t in enumerate(m.scheduler.timesteps):
            uncond_embeddings = []
            for latent in latents:
                pil_image = render_latents(latent.unsqueeze(0), m.vae)
                image_embed = clip_embed_image(pil_image, m)
                ig_spans, _ = ignorable_spans(prompt, text_embed, image_embed, span_embeds, m)
                if i < 5 or len(ig_spans) == 0:
                    uncond_embeddings.append(uncond_embedding)
                else:
                    comp_input = m.tokenizer(
                        [ig_spans[0]['c-string']], padding="max_length", max_length=max_length, return_tensors="pt"
                    )
                    comp_embedding = m.text_encoder(comp_input.input_ids.to(m.device))[0]
                    uncond_embeddings.append(comp_embedding)
            uncond_embeddings = torch.cat(uncond_embeddings, dim=0)
            text_embeddings = torch.cat([uncond_embeddings, prompt_embeddings])

            latent_model_input = torch.cat([latents]*2) if do_classifier_free_guidance else latents
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                U, C = noise_pred.chunk(2)
                stats.append((U-C).abs().sum((1, 2, 3)))
                s = guidance_scale
                noise_pred = U + s*(C-U)
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed, extra=cossim(U, C)))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True), stats


def clip_pos_batch(prompt, seeds, steps, m, guidance_scale=7.5, size=(512, 512), batch_size=1):
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    batch = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        stats = []
        batch.append(seed)
        if s_idx == len(seeds)-1:
            batch_size = len(batch)
        elif len(batch) < batch_size:
            continue

        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        
        latent_height, latent_width = height // 8, width // 8
        in_channels = m.unet.in_channels

        init_latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            init_latents = init_latents * m.scheduler.sigmas[0]

        max_length = MAX_LENGTH
        # get prompt text embeddings
        prompt_input = m.tokenizer(
            [prompt],  padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embedding = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
        text_embed = clip_embed_text(prompt, m)
        span_embeds = span_embeddings(prompt, m)
        n = len(m.tokenizer(prompt)['input_ids'])
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            text_embeddings = prompt_embeddings
        else:
            uncond_input = m.tokenizer(
                [""], padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embedding = m.text_encoder(uncond_input.input_ids.to(m.device))[0]

        latents = init_latents
        for i, t in enumerate(m.scheduler.timesteps):
            uncond_embeddings = []
            cond_embeddings = []
            for latent in latents:
                pil_image = render_latents(latent.unsqueeze(0), m.vae)
                image_embed = clip_embed_image(pil_image, m)
                ig_spans, _ = ignorable_spans(prompt, text_embed, image_embed, span_embeds, m)
                if i <= 5 or len(ig_spans) == 0:
                    uncond_embeddings.append(uncond_embedding)
                    cond_embeddings.append(prompt_embedding)
                else:
                    comp_input = m.tokenizer(
                        [ig_spans[0]['string']], padding="max_length", max_length=max_length, return_tensors="pt"
                    )
                    comp_embedding = m.text_encoder(comp_input.input_ids.to(m.device))[0]
                    uncond_embeddings.append(prompt_embedding)
                    cond_embeddings.append(comp_embedding)
            uncond_embeddings = torch.cat(uncond_embeddings, dim=0)
            cond_embeddings = torch.cat(cond_embeddings, dim=0)
            text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

            latent_model_input = torch.cat([latents]*2) if do_classifier_free_guidance else latents
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                U, C = noise_pred.chunk(2)
                stats.append((U-C).abs().sum((1, 2, 3)))
                s = guidance_scale
                noise_pred = U + s*(C-U)
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed, extra=cossim(U, C)))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True), stats

def shuffstab_batch(prompt, seeds, steps, n_shuff, m, guidance_scale=7.5, size=(512, 512), batch_size=1):
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    batch = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        stats = []
        batch.append(seed)
        if s_idx == len(seeds)-1:
            batch_size = len(batch)
        elif len(batch) < batch_size:
            continue

        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        
        latent_height, latent_width = height // 8, width // 8
        in_channels = m.unet.in_channels

        init_latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            init_latents = init_latents * m.scheduler.sigmas[0]

        max_length = MAX_LENGTH
        # get prompt text embeddings
        prompt_input = m.tokenizer(
            [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            text_embeddings = prompt_embeddings
        else:
            stabilizers = []
            words = prompt.split()
            for stab_seed in range(n_shuff):
                stab = ' '.join(random.Random(stab_seed).sample(words, k=len(words) // 3))
                stab_input = m.tokenizer(
                    [stab] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                )
                stab_embedding = m.text_encoder(stab_input.input_ids.to(m.device))[0]
                stabilizers.append(stab_embedding)
            stab_embedding = torch.cat(stabilizers, 0).mean(0, keepdim=True).repeat(batch_size, 1, 1)
            uncond_input = m.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = m.text_encoder(uncond_input.input_ids.to(m.device))[0]
            text_embeddings = torch.cat([stab_embedding, uncond_embeddings, prompt_embeddings])
            print(text_embeddings.size())

        latents = init_latents
        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*3) if do_classifier_free_guidance else latents
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                S, U, C = noise_pred.chunk(3)
                stats.append((U-C).abs().sum((1, 2, 3)))
                s = guidance_scale
                noise_pred = U + s*(C-S)
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed, extra=cossim(U, C)))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True), stats

def rf_batch(prompt, seeds, steps, m, guidance_scale=7.5, size=(512, 512), batch_size=1):
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    batch = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        stats = []
        batch.append(seed)
        if s_idx == len(seeds)-1:
            batch_size = len(batch)
        elif len(batch) < batch_size:
            continue

        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        
        latent_height, latent_width = height // 8, width // 8
        in_channels = m.unet.in_channels

        init_latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            init_latents = init_latents * m.scheduler.sigmas[0]

        max_length = MAX_LENGTH
        # get prompt text embeddings
        prompt_input = m.tokenizer(
            [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            text_embeddings = prompt_embeddings
        else:
            uncond_input = m.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = m.text_encoder(uncond_input.input_ids.to(m.device))[0]
            counter = ' '.join(prompt.split()[::-1])
            counter_input = m.tokenizer(
                [counter] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            counter_embeddings = m.text_encoder(counter_input.input_ids.to(m.device))[0]
            text_embeddings = torch.cat([uncond_embeddings, counter_embeddings, prompt_embeddings])

        latents = init_latents
        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*3) if do_classifier_free_guidance else latents
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                U, R, C = noise_pred.chunk(3)
                s = guidance_scale
                noise_pred = R + s*(C-R)
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed, extra=None))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True), stats

def c_batch(prompt, seeds, steps, m, guidance_scale=7.5, size=(512, 512), batch_size=1):
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    batch = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        stats = []
        batch.append(seed)
        if s_idx == len(seeds)-1:
            batch_size = len(batch)
        elif len(batch) < batch_size:
            continue

        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        
        latent_height, latent_width = height // 8, width // 8
        in_channels = m.unet.in_channels

        init_latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            init_latents = init_latents * m.scheduler.sigmas[0]

        max_length = MAX_LENGTH
        # get prompt text embeddings
        prompt_input = m.tokenizer(
            [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            text_embeddings = prompt_embeddings
        else:
            uncond_input = m.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = m.text_encoder(uncond_input.input_ids.to(m.device))[0]
            text_embeddings = torch.cat([uncond_embeddings, prompt_embeddings])

        latents = init_latents
        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*2) if do_classifier_free_guidance else latents
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                U, C = noise_pred.chunk(2)
                stats.append((U-C).abs().sum((1, 2, 3)))
                s = guidance_scale
                if i < 30:
                    noise_pred = C + s*(C-U)
                else:
                    noise_pred = U + s*(C-U)
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed, extra=cossim(U, C)))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True), stats

def hmmmmmmm_batch(prompt, seeds, steps, m, guidance_scale=7.5, size=(512, 512), batch_size=1):
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    batch = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        stats = []
        batch.append(seed)
        if s_idx == len(seeds)-1:
            batch_size = len(batch)
        elif len(batch) < batch_size:
            continue

        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        
        latent_height, latent_width = height // 8, width // 8
        in_channels = m.unet.in_channels

        init_latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            init_latents = init_latents * m.scheduler.sigmas[0]

        max_length = MAX_LENGTH
        # get prompt text embeddings
        prompt_input = m.tokenizer(
            [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            text_embeddings = prompt_embeddings
        else:
            uncond_input = m.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = m.text_encoder(uncond_input.input_ids.to(m.device))[0]
            n_parts = 2
            words = prompt.split()
            part_length = math.ceil(len(words) / n_parts)
            sub_prompts = [ ' '.join(words[i*part_length:min((i+1)*part_length, len(words))]) for i in range(n_parts) ]
            print(sub_prompts)
            part_embeddings = []
            for sub_prompt in sub_prompts:
                part_input = m.tokenizer(
                    [sub_prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                )
                part_embeddings.append(m.text_encoder(part_input.input_ids.to(m.device))[0])
            text_embeddings = torch.cat([uncond_embeddings, prompt_embeddings] + part_embeddings)

        latents = init_latents
        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*(2+n_parts)) if do_classifier_free_guidance else latents
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                Ns = noise_pred.chunk(2+n_parts)
                U, C = Ns[:2] 
                P = sum(Ns[2:])
                s = guidance_scale
                noise_pred = U + (s/n_parts)*(n_parts*C-P)
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed, extra=cossim(U, C)))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True), stats

def switch_batch(prompt, sub_prompts, seeds, steps, m, guidance_scale=7.5, size=(512, 512), batch_size=1):
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    batch = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        stats = []
        batch.append(seed)
        if s_idx == len(seeds)-1:
            batch_size = len(batch)
        elif len(batch) < batch_size:
            continue

        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        
        latent_height, latent_width = height // 8, width // 8
        in_channels = m.unet.in_channels

        init_latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            init_latents = init_latents * m.scheduler.sigmas[0]

        max_length = MAX_LENGTH
        # get prompt text embeddings
        prompt_input = m.tokenizer(
            [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            text_embeddings = prompt_embeddings
        else:
            uncond_input = m.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = m.text_encoder(uncond_input.input_ids.to(m.device))[0]
            n_parts = len(sub_prompts)
            print(sub_prompts)
            part_embeddings = []
            for sub_prompt in sub_prompts:
                part_input = m.tokenizer(
                    [sub_prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                )
                part_embeddings.append(m.text_encoder(part_input.input_ids.to(m.device))[0])
            text_embeddings = torch.cat([prompt_embeddings, uncond_embeddings] + part_embeddings)

        latents = init_latents
        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*(2+n_parts)) if do_classifier_free_guidance else latents
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                Ns = noise_pred.chunk(2+n_parts)
                C = Ns[0] 
                Ps = Ns[1:]
                idx = np.argmin([ (C-P).abs().sum().item() for P in Ps ])
                P = Ps[idx]
                s = guidance_scale
                noise_pred = Ps[0] + s*(C-P)
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True), stats

def bal_batch(prompt, sub_prompts, seeds, steps, m, guidance_scale=7.5, size=(512, 512), batch_size=1):
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    batch = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        stats = []
        batch.append(seed)
        if s_idx == len(seeds)-1:
            batch_size = len(batch)
        elif len(batch) < batch_size:
            continue

        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        
        latent_height, latent_width = height // 8, width // 8
        in_channels = m.unet.in_channels

        init_latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            init_latents = init_latents * m.scheduler.sigmas[0]

        max_length = MAX_LENGTH
        # get prompt text embeddings
        prompt_input = m.tokenizer(
            [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            text_embeddings = prompt_embeddings
        else:
            uncond_input = m.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = m.text_encoder(uncond_input.input_ids.to(m.device))[0]
            n_parts = len(sub_prompts)
            print(sub_prompts)
            part_embeddings = []
            for sub_prompt in sub_prompts:
                part_input = m.tokenizer(
                    [sub_prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                )
                part_embeddings.append(m.text_encoder(part_input.input_ids.to(m.device))[0])
            text_embeddings = torch.cat([uncond_embeddings, prompt_embeddings] + part_embeddings)

        latents = init_latents
        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*(2+n_parts)) if do_classifier_free_guidance else latents
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                Ns = noise_pred.chunk(2+n_parts)
                U, C = Ns[:2] 
                P = sum(Ns[2:])
                s = guidance_scale
                noise_pred = U + (s/n_parts)*(C-U + n_parts*C-P)
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed, extra=cossim(U, C)))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True), stats

def rps_batch(prompt, seeds, steps, m, guidance_scale=7.5, size=(512, 512), batch_size=1):
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    batch = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        stats = []
        batch.append(seed)
        if s_idx == len(seeds)-1:
            batch_size = len(batch)
        elif len(batch) < batch_size:
            continue

        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        
        latent_height, latent_width = height // 8, width // 8
        in_channels = m.unet.in_channels

        init_latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            init_latents = init_latents * m.scheduler.sigmas[0]

        max_length = MAX_LENGTH
        # get prompt text embeddings
        prompt_input = m.tokenizer(
            [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            text_embeddings = prompt_embeddings
        else:
            uncond_input = m.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = m.text_encoder(uncond_input.input_ids.to(m.device))[0]
            text_embeddings = torch.cat([uncond_embeddings, prompt_embeddings])

        latents = init_latents
        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*2) if do_classifier_free_guidance else latents
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                U, C = noise_pred.chunk(2)
                stats.append((U-C).abs().sum((1, 2, 3)))
                s = guidance_scale
                noise_pred = U + s*(C-U)
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True), stats

def counter_batch(prompt, seeds, steps, m, guidance_scale=7.5, size=(512, 512), batch_size=1):
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    batch = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        stats = []
        batch.append(seed)
        if s_idx == len(seeds)-1:
            batch_size = len(batch)
        elif len(batch) < batch_size:
            continue

        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        
        latent_height, latent_width = height // 8, width // 8
        in_channels = m.unet.in_channels

        init_latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            init_latents = init_latents * m.scheduler.sigmas[0]

        max_length = MAX_LENGTH
        # get prompt text embeddings
        prompt_input = m.tokenizer(
            [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            text_embeddings = prompt_embeddings
        else:
            words = prompt.split()
            counter_embeddings = []
            for i in range(len(words)):
                counter_words = words[i]
                counter_input = m.tokenizer(
                    [counter_words] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                )
                counter_embeddings.append(m.text_encoder(counter_input.input_ids.to(m.device))[0])
            counter_embeddings = torch.stack(counter_embeddings, dim=0).mean(dim=0)
            text_embeddings = torch.cat([counter_embeddings, prompt_embeddings])

        latents = init_latents
        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*2) if do_classifier_free_guidance else latents
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                U, C = noise_pred.chunk(2)
                stats.append((U-C).abs().sum((1, 2, 3)))
                s = guidance_scale
                noise_pred = U + s*(C-U)
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed, extra=cossim(U, C)))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True), stats

def rps_adj_batch(prompt, seeds, steps, m, guidance_scale=7.5, size=(512, 512), batch_size=1):
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    batch = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        stats = []
        batch.append(seed)
        if s_idx == len(seeds)-1:
            batch_size = len(batch)
        elif len(batch) < batch_size:
            continue

        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        
        latent_height, latent_width = height // 8, width // 8
        in_channels = m.unet.in_channels

        init_latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            init_latents = init_latents * m.scheduler.sigmas[0]

        max_length = MAX_LENGTH
        # get prompt text embeddings
        prompt_input = m.tokenizer(
            [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            text_embeddings = prompt_embeddings
        else:
            uncond_input = m.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = m.text_encoder(uncond_input.input_ids.to(m.device))[0]
            text_embeddings = torch.cat([uncond_embeddings, prompt_embeddings])

        latents = init_latents
        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*2) if do_classifier_free_guidance else latents
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                U, C = noise_pred.chunk(2)
                diff = (U-C).abs().sum((1, 2, 3))
                stats.append(diff)
                s = (guidance_scale * diff.div(diff.mean().item()).reciprocal()).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                noise_pred = U + s*(C-U)
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed, extra=cossim(U, C)))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True), stats

def rps_rand_batch(prompt, guide, seeds, steps, m, guidance_scale=7.5, size=(512, 512), batch_size=1):
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    batch = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        stats = []
        batch.append(seed)
        if s_idx == len(seeds)-1:
            batch_size = len(batch)
        elif len(batch) < batch_size:
            continue

        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        
        latent_height, latent_width = height // 8, width // 8
        in_channels = m.unet.in_channels

        init_latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            init_latents = init_latents * m.scheduler.sigmas[0]

        max_length = MAX_LENGTH
        # get prompt prompt embeddings
        prompt_input = m.tokenizer(
            [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            prompt_embeddings = prompt_embeddings
        else:
            uncond_input = m.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = m.text_encoder(uncond_input.input_ids.to(m.device))[0]
            guide_input = m.tokenizer(
                [guide] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            guide_embeddings = m.text_encoder(guide_input.input_ids.to(m.device))[0]

        latents = init_latents
        for i, t in enumerate(m.scheduler.timesteps):
            if i > 5:
                text_embeddings = torch.cat([uncond_embeddings, prompt_embeddings])
            else:
                text_embeddings = torch.cat([uncond_embeddings, prompt_embeddings, guide_embeddings])
            latents_factor = (text_embeddings.size(0)//batch_size)
            latent_model_input = torch.cat([latents]*latents_factor) if do_classifier_free_guidance else latents
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                N = noise_pred.chunk(latents_factor)
                s = guidance_scale
                if len(N) == 2:
                    U, C = N
                    noise_pred = U + s*(C-U)
                else:
                    U, C, G = N
                    noise_pred = U + s*(C-U + C-G)
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True), stats

def ts_batch(prompt, seeds, steps, m, guidance_scale=7.5, size=(512, 512), batch_size=1):
    height, width = size

    text_embed = clip_embed_text(prompt, m)

    data = []
    batch = []
    with torch.no_grad():
      for s_idx, seed in enumerate(seeds):
        stats = []
        batch.append(seed)
        if s_idx == len(seeds)-1:
            batch_size = len(batch)
        elif len(batch) < batch_size:
            continue

        generators = [ torch.Generator(device=m.device).manual_seed(seed) for seed in batch ]
        m.scheduler.set_timesteps(steps)
        
        latent_height, latent_width = height // 8, width // 8
        in_channels = m.unet.in_channels

        init_latents = torch.cat([ torch.randn(
          (1, in_channels, latent_height, latent_width),
          generator=generator,
          device=m.device
          ) for generator in generators ], dim=0)
        if type(m.scheduler) is LMSDiscreteScheduler:
            init_latents = init_latents * m.scheduler.sigmas[0]

        max_length = MAX_LENGTH
        # get prompt text embeddings
        prompt_input = m.tokenizer(
            [prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        prompt_embeddings = m.text_encoder(prompt_input.input_ids.to(m.device))[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            text_embeddings = prompt_embeddings
        else:
            uncond_input = m.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = m.text_encoder(uncond_input.input_ids.to(m.device))[0]
            text_embeddings = torch.cat([uncond_embeddings, prompt_embeddings])

        latents = init_latents
        for i, t in enumerate(m.scheduler.timesteps):
            latent_model_input = torch.cat([latents]*2) if do_classifier_free_guidance else latents
            if type(m.scheduler) is LMSDiscreteScheduler:
                sigma = m.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            noise_pred = m.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                s = guidance_scale
                U, C = noise_pred.chunk(2)
                sign_mask = U.sign() == C.sign()
                size_mask = C.gt(U)
                comp_mask = torch.logical_and(sign_mask, size_mask)
                rev_comp_mask = torch.logical_and(sign_mask, size_mask.logical_not())
                RM = (C-U)*comp_mask + (C-U)*rev_comp_mask
                noise_pred = C + s*RM
            if type(m.scheduler) is LMSDiscreteScheduler:
                latents = m.scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = m.scheduler.step(noise_pred, t, latents)["prev_sample"]
        for seed, latent in zip(batch, latents):
            pil_image = render_latents(latent.unsqueeze(0), m.vae)
            image_embed = clip_embed_image(pil_image, m)
            score = (text_embed @ image_embed.T).item()
            data.append(Cand(score, pil_image, seed, prompt, text_embed, image_embed))
        batch = []
    return sorted(data, key=lambda c: c.score, reverse=True), stats


