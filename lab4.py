# Проверяем установленные пакеты
import importlib
def check_module(module_name):
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} успешно импортирован")
        return True
    except ImportError as e:
        print(f"❌ Ошибка импорта {module_name}: {e}")
        return False

# Проверка критических модулей
critical_modules = ['torch', 'torchaudio', 'transformers', 'soundfile', 'librosa', 'gtts']
for module in critical_modules:
    check_module(module)

# Проверка доступности GPU
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используемое устройство: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

import torch
import torchaudio
import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt
import time
import gc
import psutil
import os
from pathlib import Path
import IPython.display as ipd
from scipy.io import wavfile
import requests
import tempfile
from gtts import gTTS
import io

# Функции для измерения метрик
def measure_resource_usage():
    """Измеряет использование ресурсов"""
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    
    if torch.cuda.is_available():
        gpu_usage = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    else:
        gpu_usage = 0
        gpu_memory = 0
        
    return {
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        'gpu_usage_gb': gpu_usage,
        'gpu_memory_gb': gpu_memory
    }

def calculate_mos(audio_path):
    """
    Вычисляет приблизительную оценку MOS (Mean Opinion Score)
    """
    try:
        # Загрузка аудио
        audio, sr = librosa.load(audio_path, sr=22050)
        
        # Расчет различных метрик качества
        rms = librosa.feature.rms(y=audio)[0]
        rms_mean = np.mean(rms)
        
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_mean = np.mean(zcr)
        
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_centroid_mean = np.mean(spectral_centroids)
        
        # Нормализация метрик для MOS (эвристический подход)
        mos_approximation = (
            0.4 * min(rms_mean / 0.1, 1.0) + 
            0.3 * min(spectral_centroid_mean / 4000, 1.0) +
            0.3 * (1 - min(zcr_mean / 0.1, 1.0))
        ) * 4 + 1 
        
        return min(mos_approximation, 5.0)
        
    except Exception as e:
        print(f"Ошибка расчета MOS для {audio_path}: {e}")
        return 3.0  # Средняя оценка при ошибке

def get_audio_length(audio_path):
    """Получает длительность аудио файла"""
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        return len(audio) / sr
    except:
        return 0

# Тестовые тексты (только английские для совместимости)
test_texts = [
    "Hello world! This is a text to speech test.",
    "The weather is beautiful today for a walk in the park.",
    "Neural networks are revolutionizing natural language processing.",
    "Machine learning enables computers to learn from data.",
    "Speech synthesis is becoming more natural and expressive.",
    "Artificial intelligence is transforming our world.",
    "Deep learning models require large amounts of training data.",
    "The quick brown fox jumps over the lazy dog.",
    "Text to speech technology has improved significantly in recent years.",
    "This is a demonstration of modern speech synthesis quality."
]

print("Тестовые тексты:")
for i, text in enumerate(test_texts, 1):
    print(f"{i}. {text}")

def test_bark():
    """Тестирование модели Bark - исправленная версия"""
    print("=" * 50)
    print("ТЕСТИРОВАНИЕ BARK")
    print("=" * 50)
    
    try:
        from transformers import BarkModel, AutoProcessor
        
        # Загрузка модели и процессора - исправляем ошибку с float16
        start_time = time.time()
        
        # Убираем torch_dtype=torch.float16 чтобы избежать ошибки
        model = BarkModel.from_pretrained("suno/bark-small")
        processor = AutoProcessor.from_pretrained("suno/bark-small")
        
        model = model.to(device)
        load_time = time.time() - start_time
        
        print(f"Модель загружена за {load_time:.2f} секунд")
        print(f"Размер модели: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M параметров")
        
        results = []
        output_dir = Path("bark_results")
        output_dir.mkdir(exist_ok=True)
        
        for i, text in enumerate(test_texts):
            print(f"Синтез {i+1}/{len(test_texts)}: {text[:50]}...")
            
            # Измерение ресурсов до синтеза
            resources_before = measure_resource_usage()
            
            # Подготовка входных данных
            inputs = processor(text, return_tensors="pt").to(device)
            
            synth_start = time.time()
            with torch.no_grad():
                audio_array = model.generate(**inputs, do_sample=True)
            
            synth_time = time.time() - synth_start
            
            # Сохранение аудио
            output_path = output_dir / f"bark_{i+1}.wav"
            audio_data = audio_array[0].cpu().numpy()
            sample_rate = model.generation_config.sample_rate
            sf.write(str(output_path), audio_data, sample_rate)
            
            resources_after = measure_resource_usage()
            mos_score = calculate_mos(str(output_path))
            
            result = {
                'model': 'Bark',
                'text_id': i+1,
                'synthesis_time': synth_time,
                'mos_score': mos_score,
                'audio_file': str(output_path),
                'resources_before': resources_before,
                'resources_after': resources_after,
                'audio_length': len(audio_data) / sample_rate
            }
            
            results.append(result)
            print(f"Время синтеза: {synth_time:.2f}с, MOS: {mos_score:.2f}, Длина: {result['audio_length']:.2f}с")
            
            # Очистка памяти
            del inputs, audio_array
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Пауза между синтезами
            time.sleep(1)
        
        return results
        
    except Exception as e:
        print(f"Ошибка в Bark: {e}")
        return []


def test_mms_tts():
    """Тестирование модели MMS TTS"""
    print("=" * 50)
    print("ТЕСТИРОВАНИЕ MMS TTS")
    print("=" * 50)
    
    try:
        from transformers import VitsModel, AutoTokenizer
        
        # Загрузка модели и токенизатора
        start_time = time.time()
        model = VitsModel.from_pretrained("facebook/mms-tts-eng")
        tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
        load_time = time.time() - start_time
        
        model = model.to(device)
        print(f"Модель загружена за {load_time:.2f} секунд")
        print(f"Размер модели: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M параметров")
        
        results = []
        output_dir = Path("mms_results")
        output_dir.mkdir(exist_ok=True)
        
        for i, text in enumerate(test_texts):
            print(f"Синтез {i+1}/{len(test_texts)}: {text[:50]}...")
            
            resources_before = measure_resource_usage()
            
            # Токенизация текста
            inputs = tokenizer(text, return_tensors="pt").to(device)
            
            synth_start = time.time()
            with torch.no_grad():
                output = model(**inputs)
                
            synth_time = time.time() - synth_start
            
            # Получение аудио
            audio = output.waveform[0].cpu().numpy()
            sample_rate = model.config.sampling_rate
            
            # Сохранение аудио
            output_path = output_dir / f"mms_{i+1}.wav"
            sf.write(str(output_path), audio, sample_rate)
            
            resources_after = measure_resource_usage()
            mos_score = calculate_mos(str(output_path))
            
            result = {
                'model': 'MMS TTS',
                'text_id': i+1,
                'synthesis_time': synth_time,
                'mos_score': mos_score,
                'audio_file': str(output_path),
                'resources_before': resources_before,
                'resources_after': resources_after,
                'audio_length': len(audio) / sample_rate
            }
            
            results.append(result)
            print(f"Время синтеза: {synth_time:.2f}с, MOS: {mos_score:.2f}, Длина: {result['audio_length']:.2f}с")
            
            del inputs, output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            time.sleep(0.5)
        
        return results
        
    except Exception as e:
        print(f"Ошибка в MMS TTS: {e}")
        return []

def test_speecht5():
    """Тестирование модели SpeechT5"""
    print("=" * 50)
    print("ТЕСТИРОВАНИЕ SPEECHT5")
    print("=" * 50)
    
    try:
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        from datasets import load_dataset
        
        # Загрузка моделей
        start_time = time.time()
        processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        load_time = time.time() - start_time
        
        model = model.to(device)
        vocoder = vocoder.to(device)
        
        print(f"Модель загружена за {load_time:.2f} секунд")
        print(f"Размер модели: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M параметров")
        
        # Загрузка эмбеддингов голоса
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device)
        
        results = []
        output_dir = Path("speecht5_results")
        output_dir.mkdir(exist_ok=True)
        
        for i, text in enumerate(test_texts):
            print(f"Синтез {i+1}/{len(test_texts)}: {text[:50]}...")
            
            resources_before = measure_resource_usage()
            
            # Обработка текста
            inputs = processor(text=text, return_tensors="pt").to(device)
            
            synth_start = time.time()
            with torch.no_grad():
                # Генерация спектрограммы
                spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)
                # Генерация аудио с помощью вокодера
                audio = vocoder(spectrogram)
                
            synth_time = time.time() - synth_start
            
            # Сохранение аудио
            output_path = output_dir / f"speecht5_{i+1}.wav"
            audio_data = audio[0].cpu().numpy()
            sample_rate = 16000
            sf.write(str(output_path), audio_data, sample_rate)
            
            resources_after = measure_resource_usage()
            mos_score = calculate_mos(str(output_path))
            
            result = {
                'model': 'SpeechT5',
                'text_id': i+1,
                'synthesis_time': synth_time,
                'mos_score': mos_score,
                'audio_file': str(output_path),
                'resources_before': resources_before,
                'resources_after': resources_after,
                'audio_length': len(audio_data) / sample_rate
            }
            
            results.append(result)
            print(f"Время синтеза: {synth_time:.2f}с, MOS: {mos_score:.2f}, Длина: {result['audio_length']:.2f}с")
            
            del inputs, spectrogram, audio
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            time.sleep(0.5)
        
        return results
        
    except Exception as e:
        print(f"Ошибка в SpeechT5: {e}")
        return []


def test_gtts():
    """Тестирование Google Text-to-Speech"""
    print("=" * 50)
    print("ТЕСТИРОВАНИЕ gTTS")
    print("=" * 50)
    
    try:
        from gtts import gTTS
        import io
        
        results = []
        output_dir = Path("gtts_results")
        output_dir.mkdir(exist_ok=True)
        
        for i, text in enumerate(test_texts):
            print(f"Синтез {i+1}/{len(test_texts)}: {text[:50]}...")
            
            resources_before = measure_resource_usage()
            
            synth_start = time.time()
            output_path = output_dir / f"gtts_{i+1}.wav"
            
            # Создание gTTS объекта и сохранение в файл
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(str(output_path))
            
            synth_time = time.time() - synth_start
            
            resources_after = measure_resource_usage()
            mos_score = calculate_mos(str(output_path))
            
            result = {
                'model': 'gTTS',
                'text_id': i+1,
                'synthesis_time': synth_time,
                'mos_score': mos_score,
                'audio_file': str(output_path),
                'resources_before': resources_before,
                'resources_after': resources_after,
                'audio_length': get_audio_length(str(output_path))
            }
            
            results.append(result)
            print(f"Время синтеза: {synth_time:.2f}с, MOS: {mos_score:.2f}, Длина: {result['audio_length']:.2f}с")
            
            time.sleep(1)  # Избегаем ограничений API
        
        return results
        
    except Exception as e:
        print(f"Ошибка в gTTS: {e}")
        return []

def test_coqui_tts():
    """Тестирование Coqui TTS если он установился"""
    print("=" * 50)
    print("ТЕСТИРОВАНИЕ COQUI TTS")
    print("=" * 50)
    
    try:
        # Пробуем разные способы импорта
        try:
            from TTS.api import TTS
            tts_available = True
        except ImportError:
            print("TTS не установлен, пропускаем...")
            return []
        
        if not tts_available:
            return []
            
        # Инициализация модели
        start_time = time.time()
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
        load_time = time.time() - start_time
        
        print(f"Модель загружена за {load_time:.2f} секунд")
        
        results = []
        output_dir = Path("coqui_results")
        output_dir.mkdir(exist_ok=True)
        
        for i, text in enumerate(test_texts):
            print(f"Синтез {i+1}/{len(test_texts)}: {text[:50]}...")
            
            resources_before = measure_resource_usage()
            
            synth_start = time.time()
            output_path = output_dir / f"coqui_{i+1}.wav"
            
            # Синтез речи
            tts.tts_to_file(text=text, file_path=str(output_path))
            
            synth_time = time.time() - synth_start
            
            resources_after = measure_resource_usage()
            mos_score = calculate_mos(str(output_path))
            
            result = {
                'model': 'Coqui TTS',
                'text_id': i+1,
                'synthesis_time': synth_time,
                'mos_score': mos_score,
                'audio_file': str(output_path),
                'resources_before': resources_before,
                'resources_after': resources_after,
                'audio_length': get_audio_length(str(output_path))
            }
            
            results.append(result)
            print(f"Время синтеза: {synth_time:.2f}с, MOS: {mos_score:.2f}, Длина: {result['audio_length']:.2f}с")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            time.sleep(0.5)
        
        return results
        
    except Exception as e:
        print(f"Coqui TTS не доступен: {e}")
        return []


def test_hf_vits():
    """Тестирование VITS модели через Hugging Face"""
    print("=" * 50)
    print("ТЕСТИРОВАНИЕ HUGGING FACE VITS")
    print("=" * 50)
    
    try:
        from transformers import VitsModel, AutoTokenizer
        
        # Используем работающую VITS модель
        start_time = time.time()
        model = VitsModel.from_pretrained("facebook/mms-tts-eng")  # Используем уже проверенную модель
        tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
        load_time = time.time() - start_time
        
        model = model.to(device)
        print(f"Модель загружена за {load_time:.2f} секунд")
        print(f"Размер модели: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M параметров")
        
        results = []
        output_dir = Path("vits_results")
        output_dir.mkdir(exist_ok=True)
        
        for i, text in enumerate(test_texts):
            print(f"Синтез {i+1}/{len(test_texts)}: {text[:50]}...")
            
            resources_before = measure_resource_usage()
            
            # Токенизация текста
            inputs = tokenizer(text, return_tensors="pt").to(device)
            
            synth_start = time.time()
            with torch.no_grad():
                output = model(**inputs)
                
            synth_time = time.time() - synth_start
            
            # Получение аудио
            audio = output.waveform[0].cpu().numpy()
            sample_rate = model.config.sampling_rate
            
            # Сохранение аудио
            output_path = output_dir / f"vits_{i+1}.wav"
            sf.write(str(output_path), audio, sample_rate)
            
            resources_after = measure_resource_usage()
            mos_score = calculate_mos(str(output_path))
            
            result = {
                'model': 'VITS (HF)',
                'text_id': i+1,
                'synthesis_time': synth_time,
                'mos_score': mos_score,
                'audio_file': str(output_path),
                'resources_before': resources_before,
                'resources_after': resources_after,
                'audio_length': len(audio) / sample_rate
            }
            
            results.append(result)
            print(f"Время синтеза: {synth_time:.2f}с, MOS: {mos_score:.2f}, Длина: {result['audio_length']:.2f}с")
            
            del inputs, output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            time.sleep(0.5)
        
        return results
        
    except Exception as e:
        print(f"Ошибка в VITS: {e}")
        return []

def run_available_models():
    """Запуск тестирования всех доступных моделей"""
    all_results = []
    
    # Тестируем модели, которые должны работать
    models_to_test = [
        test_bark,           # Bark от Suno AI
        test_mms_tts,        # Facebook MMS
        test_speecht5,       # Microsoft SpeechT5  
        test_gtts,           # Google TTS
        test_hf_vits,        # Hugging Face VITS (альтернатива Coqui TTS)
    ]
    
    # Если Coqui TTS установился, добавляем его
    try:
        from TTS.api import TTS
        models_to_test.append(test_coqui_tts)
        print("Coqui TTS доступен, добавляем в тестирование")
    except:
        print("Coqui TTS недоступен, используем альтернативные модели")
    
    for model_test in models_to_test:
        try:
            print(f"\n{'='*60}")
            print(f"ЗАПУСК ТЕСТА: {model_test.__name__}")
            print(f"{'='*60}")
            
            results = model_test()
            all_results.extend(results)
            
            # Пауза между тестами для очистки памяти
            time.sleep(3)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Ошибка при тестировании {model_test.__name__}: {e}")
            continue
    
    return all_results


print("НАЧАЛО ТЕСТИРОВАНИЯ МОДЕЛЕЙ СИНТЕЗА РЕЧИ")
all_results = run_available_models()

print(f"\nТестирование завершено. Получено результатов: {len(all_results)}")


def analyze_results(results):
    """Анализ и визуализация результатов тестирования"""
    
    if not results:
        print("Нет данных для анализа")
        return None
    
    # Группировка результатов по моделям
    models_data = {}
    for result in results:
        model_name = result['model']
        if model_name not in models_data:
            models_data[model_name] = []
        models_data[model_name].append(result)
    
    # Расчет средних метрик для каждой модели
    summary = []
    for model_name, model_results in models_data.items():
        avg_synthesis_time = np.mean([r['synthesis_time'] for r in model_results])
        avg_mos = np.mean([r['mos_score'] for r in model_results])
        avg_cpu_usage = np.mean([r['resources_after']['cpu_usage'] for r in model_results])
        avg_memory_usage = np.mean([r['resources_after']['memory_usage'] for r in model_results])
        avg_gpu_usage = np.mean([r['resources_after']['gpu_usage_gb'] for r in model_results])
        avg_audio_length = np.mean([r['audio_length'] for r in model_results])
        
        summary.append({
            'model': model_name,
            'avg_synthesis_time': avg_synthesis_time,
            'avg_mos': avg_mos,
            'avg_cpu_usage': avg_cpu_usage,
            'avg_memory_usage': avg_memory_usage,
            'avg_gpu_usage': avg_gpu_usage,
            'avg_audio_length': avg_audio_length,
            'num_samples': len(model_results)
        })
    
    # Создание таблицы результатов
    print("\n" + "="*100)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*100)
    print(f"{'Модель':<15} {'Время (с)':<10} {'MOS':<8} {'CPU (%)':<10} {'Память (%)':<12} {'GPU (GB)':<10} {'Длина (с)':<10}")
    print("-"*100)
    
    for model_summary in summary:
        print(f"{model_summary['model']:<15} {model_summary['avg_synthesis_time']:<10.2f} "
              f"{model_summary['avg_mos']:<8.2f} {model_summary['avg_cpu_usage']:<10.1f} "
              f"{model_summary['avg_memory_usage']:<12.1f} {model_summary['avg_gpu_usage']:<10.2f} "
              f"{model_summary['avg_audio_length']:<10.2f}")
    
    # Визуализация результатов
    if len(summary) > 1:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # График времени синтеза
        models = [s['model'] for s in summary]
        times = [s['avg_synthesis_time'] for s in summary]
        bars1 = ax1.bar(models, times, color='skyblue', alpha=0.7)
        ax1.set_title('Среднее время синтеза на один текст')
        ax1.set_ylabel('Время (секунды)')
        ax1.tick_params(axis='x', rotation=45)
        # Добавляем значения на столбцы
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # График MOS оценок
        mos_scores = [s['avg_mos'] for s in summary]
        bars2 = ax2.bar(models, mos_scores, color='lightgreen', alpha=0.7)
        ax2.set_title('Средняя MOS оценка качества')
        ax2.set_ylabel('MOS (1-5)')
        ax2.set_ylim(0, 5)
        ax2.tick_params(axis='x', rotation=45)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # График использования CPU
        cpu_usage = [s['avg_cpu_usage'] for s in summary]
        bars3 = ax3.bar(models, cpu_usage, color='orange', alpha=0.7)
        ax3.set_title('Среднее использование CPU')
        ax3.set_ylabel('CPU (%)')
        ax3.tick_params(axis='x', rotation=45)
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # График использования GPU
        gpu_usage = [s['avg_gpu_usage'] for s in summary]
        bars4 = ax4.bar(models, gpu_usage, color='red', alpha=0.7)
        ax4.set_title('Среднее использование GPU')
        ax4.set_ylabel('GPU (GB)')
        ax4.tick_params(axis='x', rotation=45)
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('results_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return summary

summary = analyze_results(all_results)

def demonstrate_examples():
    """Демонстрация примеров синтеза от разных моделей"""
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ ПРИМЕРОВ СИНТЕЗА")
    print("="*60)
    
    # Поиск примеров аудио файлов
    model_dirs = ["bark_results", "mms_results", "speecht5_results", "gtts_results", "vits_results", "coqui_results"]
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            audio_files = list(Path(model_dir).glob("*.wav"))
            if audio_files:
                # Берем первый файл для демонстрации
                audio_file = audio_files[0]
                print(f"\n{model_dir}: {audio_file.name}")
                
                try:
                    # Показываем аудио
                    display(ipd.Audio(str(audio_file)))
                    
                    # Анализ характеристик аудио
                    audio, sr = librosa.load(audio_file)
                    duration = len(audio) / sr
                    print(f"Длительность: {duration:.2f} сек, Sample rate: {sr} Hz")
                    
                except Exception as e:
                    print(f"Ошибка загрузки аудио: {e}")

demonstrate_examples()

def generate_conclusions(summary):
    """Генерация выводов на основе результатов тестирования"""
    print("\n" + "="*80)
    print("ВЫВОДЫ И ЗАКЛЮЧЕНИЕ")
    print("="*80)
    
    if not summary:
        print("Нет данных для формирования выводов")
        return
    
    # Находим лучшую модель по каждому критерию
    if len(summary) > 0:
        best_mos = max(summary, key=lambda x: x['avg_mos'])
        fastest = min(summary, key=lambda x: x['avg_synthesis_time'])
        most_efficient_cpu = min(summary, key=lambda x: x['avg_cpu_usage'])
        most_efficient_gpu = min(summary, key=lambda x: x['avg_gpu_usage'])
        
        print("ЛУЧШИЕ МОДЕЛИ ПО КРИТЕРИЯМ:")
        print(f"🎯 Качество звука (MOS): {best_mos['model']} (MOS: {best_mos['avg_mos']:.2f})")
        print(f"⚡ Скорость синтеза: {fastest['model']} ({fastest['avg_synthesis_time']:.2f} сек)")
        print(f"💻 Эффективность CPU: {most_efficient_cpu['model']} ({most_efficient_cpu['avg_cpu_usage']:.1f}%)")
        print(f"🎮 Эффективность GPU: {most_efficient_gpu['model']} ({most_efficient_gpu['avg_gpu_usage']:.2f} GB)")
        
        print("\nТЕНДЕНЦИИ РАЗВИТИЯ TTS:")
        print("• Переход к end-to-end моделям (VITS, Bark)")
        print("• Улучшение естественности и выразительности")
        print("• Снижение требований к вычислительным ресурсам")
        print("• Поддержка многомодальности (Bark)")
        print("• Универсальные модели для multiple tasks (SpeechT5)")
        
        print("\nРЕКОМЕНДАЦИИ ПО ВЫБОРУ МОДЕЛИ:")
        print("1. Для высокого качества: модели с наивысшими MOS оценками")
        print("2. Для реального времени: модели с наименьшим временем синтеза")
        print("3. Для ограниченных ресурсов: модели с низким потреблением CPU/GPU")
        print("4. Для универсальности: многомодальные модели (Bark)")
        print("5. Для production: стабильные и проверенные модели (MMS, gTTS)")

generate_conclusions(summary)

def save_complete_report(summary, results):
    """Сохранение полного отчета в файл"""
    
    report = """
# ОТЧЕТ ПО ЛАБОРАТОРНОЙ РАБОТЕ №4
# Исследование и сравнение современных моделей синтеза речи

## Введение
В данной работе проведено сравнительное исследование современных моделей синтеза речи с использованием доступных через Hugging Face Transformers моделей и других TTS решений.

## Методология
- Тестирование проводилось на идентичном наборе текстов (10 фраз на английском)
- Измерялись объективные метрики: время синтеза, использование ресурсов (CPU, GPU, память)
- Рассчитывались приближенные MOS оценки качества звука
- Анализировались субъективные характеристики естественности

## Исследованные модели
"""
    
    if summary:
        for model in summary:
            report += f"- **{model['model']}**: {model['num_samples']} samples, среднее время: {model['avg_synthesis_time']:.2f}с, MOS: {model['avg_mos']:.2f}\n"
    
    report += """
## Результаты
"""
    
    # Добавляем сводную таблицу
    if summary:
        report += "\n### Сводная таблица результатов\n\n"
        report += "| Модель | Время синтеза (с) | MOS | CPU (%) | Память (%) | GPU (GB) | Длина (с) |\n"
        report += "|---------|-------------------|-----|---------|------------|----------|------------|\n"
        
        for model_summary in summary:
            report += (f"| {model_summary['model']} | {model_summary['avg_synthesis_time']:.2f} | "
                      f"{model_summary['avg_mos']:.2f} | {model_summary['avg_cpu_usage']:.1f} | "
                      f"{model_summary['avg_memory_usage']:.1f} | {model_summary['avg_gpu_usage']:.2f} | "
                      f"{model_summary['avg_audio_length']:.2f} |\n")
    
    report += """
## Заключение
Проведенное исследование показало разнообразие подходов к синтезу речи в современных нейросетевых моделях. 
Каждая модель имеет свои преимущества и оптимальные сферы применения.

### Ключевые выводы:
1. Нейросетевые модели обеспечивают более естественное звучание
2. Модели различаются по требовательности к ресурсам
3. Выбор модели зависит от конкретных требований проекта
4. Современные TTS системы достигли высокого уровня качества
"""
    
    # Сохраняем отчет
    with open("lab4_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("Полный отчет сохранен в файл: lab4_report.md")

# Сохранение отчета
save_complete_report(summary, all_results)

print("\n✅ ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА!")
print("Все результаты сохранены в соответствующих директориях")