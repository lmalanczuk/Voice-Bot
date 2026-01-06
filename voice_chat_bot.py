import os
import time
import speech_recognition as sr
import pyttsx3
from faster_whisper import WhisperModel
from llama_cpp import Llama
import multiprocessing as mp


MODEL_PATH = "bielik/bielik-7b-instruct-v0.1.Q4_K_M.gguf"
WORKSPACE_DIR = "assistant_workspace"
os.makedirs(WORKSPACE_DIR, exist_ok=True)


class VoiceAgent:
    def __init__(self, model_path):
        print("Inicjalizacja Whisper")
        self.whisper = WhisperModel("small", device="cpu", compute_type="int8")

        print(f"Ładowanie LLM ({model_path})...")
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_gpu_layers=1,
                verbose=False
            )
        except Exception as e:
            print(f"Krytyczny błąd modelu: {e}")
            exit(1)

        self.history = []
        self.system_prompt = (
            "Jesteś inteligentnym asystentem głosowym. Odpowiadaj zwięźle i naturalnie. "
            "Jeśli użytkownik poprosi o zapisanie czegoś, zakończ swoją wypowiedź specjalną komendą: "
            "[[NOTE: treść notatki]]. Nie pytaj czy zapisać, po prostu użyj komendy."
        )


        self.voice_id = self._find_polish_voice_id()

    def _find_polish_voice_id(self):
        try:
            temp_engine = pyttsx3.init()
            voices = temp_engine.getProperty('voices')
            found_id = None
            for voice in voices:
                if "pl" in voice.language or "Paulina" in voice.language:
                    found_id = voice.id
                    break
            del temp_engine
            return found_id
        except Exception as e:
            print(f"błąd konfiguracji głosu: {e}")


    def speak(self, text):
        if not text: return

        clean_text = text.split("[[")[0].strip()
        if not clean_text: return

        print(f"Asystent: {clean_text}")

        engine = None
        try:
            engine = pyttsx3.init()

            if self.voice_id:
                engine.setProperty('voice', self.voice_id)
            engine.setProperty('rate', 160)

            engine.say(text)
            engine.runAndWait()

        except Exception as e:
            print(f"Błąd mowy: {e}")

    def listen(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("\nSłucham... (Mów teraz)")
            try:
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.listen(source, timeout=3, phrase_time_limit=8)

                with open("temp.wav", "wb") as f:
                    f.write(audio.get_wav_data())

                segments, _ = self.whisper.transcribe("temp.wav", language="pl")
                text = "".join([s.text for s in segments]).strip()

                if text:
                    print(f"Użytkownik: {text}")
                    return text
                return None

            except sr.WaitTimeoutError:
                print("... Cisza ...")
                return None
            except Exception as e:
                print(f"Błąd mikrofonu: {e}")
                return None

    def think(self, user_input):
        self.history.append(f"Użytkownik: {user_input}")

        conversation_context = "\n".join(self.history[-6:])
        full_prompt = f"System: {self.system_prompt}\n\nKonwersacja:\n{conversation_context}\nAsystent:"

        output = self.llm(
            full_prompt,
            max_tokens=256,
            stop=["Użytkownik:", "System:"],
            echo=False,
            temperature=0.7
        )
        response_text = output['choices'][0]['text'].strip()

        self.history.append(f"Asystent: {response_text}")

        return response_text

    def execute_tools(self, response_text):

        if "[[NOTE:" in response_text:
            try:
                start = response_text.find("[[NOTE:") + 7
                end = response_text.find("]]", start)
                note_content = response_text[start:end].strip()

                self._create_note_file(note_content)
                return True
            except Exception as e:
                print(f"Błąd parsowania narzędzia: {e}")

        return False

    def _create_note_file(self, content):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"notatka_{timestamp}.txt"
        path = os.path.join(WORKSPACE_DIR, filename)

        with open(path, "w", encoding='utf-8') as f:
            f.write(content)

        print(f"[SYSTEM] Zapisano notatkę: {filename}")

    def run(self):
        self.speak("System gotowy. Wciśnij Enter, aby rozmawiać.")

        while True:
            cmd = input("\nENTER = Mów | 'q' = Wyjście: ")
            if cmd.lower() == 'q':
                break

            user_text = self.listen()

            if user_text:
                response = self.think(user_text)

                tool_used = self.execute_tools(response)

                self.speak(response)


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Nie znaleziono modelu pod ścieżką: {MODEL_PATH}")
        print("Pobierz model GGUF (np. Bielik) i zaktualizuj zmienną MODEL_PATH.")
    else:
        agent = VoiceAgent(MODEL_PATH)
        agent.run()