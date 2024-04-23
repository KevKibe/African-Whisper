
mod multilingual;
mod pcm_decode;
mod resample;
mod decode;
mod model;

use anyhow::{Error as E, Result};
use tokenizers::Tokenizer;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::PathBuf;
use candle_transformers::models::whisper::{self as m, audio, Config};


use decode::{Decoder, Task, token_id};
use resample::AudioResampler;
use model::Model;


    



fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    // Set up tracing as desired
    let tracing_enabled = true;
    let _guard = if tracing_enabled {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    // To use or not to use GPU ~ Shakeskarparthy

    // let use_cpu = Device::cuda_if_available; 
    let use_cpu = true;     
    let device = Device::Cpu;
    println!("GPU status {}", use_cpu);
    

    let quantized = true; // true if using quantized model, false otherwise

    let model_id = "openai/whisper-tiny"; // Define model_id directly
    let revision = "main"; // Define revision directly

    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(model_id.to_string(), RepoType::Model, revision.to_string()));
    let path = "/Users/la/Desktop/Projects/African Whisper/src/deployment/rustinference/src/samples_jfk.wav";
    let buf_path = PathBuf::from(path);
    let resampler = AudioResampler::new(buf_path, 16_000);

    let output_path = match resampler.resample() {
        Ok(output_path) => {
            // If the result is Ok, print the output path
            println!("Resampled audio file saved at: {}", output_path);
            output_path // Store the output path for further use
        }
        Err(error) => {
            // If the result is Err, print an error message
            println!("Failed to resample audio: {:?}", error);
            // Convert the error to anyhow::Error and return it
            return Err(anyhow::Error::new(error));
        }
    };

    

    let (config, tokenizer, model) = (
        repo.get("config.json")?,
        repo.get("tokenizer.json")?,
        repo.get("pytorch_model.bin")?,
    );
    let config: Config = serde_json::from_str(&std::fs::read_to_string(config)?)?;
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;

    // Load mel filters based on the number of mel bins
    let mel_bytes = match config.num_mel_bins {
        80 => include_bytes!("melfilters.bytes").as_slice(),
        128 => include_bytes!("melfilters128.bytes").as_slice(),
        nmel => panic!("Unexpected num_mel_bins: {nmel}"),
    };
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);


    // // Decode PCM data
    let (pcm_data, sample_rate) = pcm_decode::pcm_decode(output_path)?;
    if sample_rate != m::SAMPLE_RATE as u32 {
        panic!("Input file must have a {} sampling rate", m::SAMPLE_RATE);
    }
    println!("PCM data loaded: {}", pcm_data.len());

    // Convert PCM data to mel
    let mel = audio::pcm_to_mel(&config, &pcm_data, &mel_filters);
    let mel_len = mel.len();
    let mel = Tensor::from_vec(
        mel,
        (1, config.num_mel_bins, mel_len / config.num_mel_bins),
        &device,
    )?;
    println!("Loaded mel: {:?}", mel.dims());

    // Load model based on whether it is quantized
    let mut model = if quantized {
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
            &model,
            &device,
        )?;
        Model::Quantized(m::quantized_model::Whisper::load(&vb, config)?)
    } else {
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model], m::DTYPE, &device)? };
        Model::Normal(m::model::Whisper::load(&vb, config)?)
    };

    // Define language token directly
    let language = Some("sw"); // Example language (English)
    let language_token = match (true, language) { // Assuming the model is multilingual
        (true, None) => Some(multilingual::detect_language(&mut model, &tokenizer, &mel)?),
        (true, Some(language)) => match token_id(&tokenizer, &format!("<|{language}|>")) {
            Ok(token_id) => Some(token_id),
            Err(_) => panic!("Language {language} is not supported"),
        },
        _ => panic!("Invalid configuration for language token"),
    };

    // Define other parameters directly
    let seed = 42; // Example seed
    let task = Some(Task::Transcribe); // Example task (transcription)
    let timestamps = false; // Example setting (no timestamps)
    let verbose = true; // Example setting (verbose mode)

    // Create a decoder and run the process
    let mut dc = Decoder::new(
        model,
        tokenizer,
        seed,
        &device,
        language_token,
        task,
        timestamps,
        verbose,
    )?;
    dc.run(&mel)?;

    Ok(())

}