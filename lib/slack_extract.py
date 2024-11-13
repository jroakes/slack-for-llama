import os
import json
from typing import Dict, List, Tuple, Any, Callable
from pathlib import Path
from tqdm.auto import tqdm

Message = Dict[str, Any]
Channel = Dict[str, Any]
User = Dict[str, Any]


class Slack:
    def __init__(self, data_dir: str):
        self.users = None
        self.data_dir = Path(data_dir)
        
        # Load users
        try:
            with open(self.data_dir / "users.json", encoding='utf-8') as f:
                self.users = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"users.json not found in {data_dir}")
            
        # Load channels
        try:
            with open(self.data_dir / "channels.json", encoding='utf-8') as f:
                self.channels = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"channels.json not found in {data_dir}")
            
        self.messages = {}
        
        # Files to skip
        skip_files = {
            "integration_logs.json",
            "users.json",
            "channels.json",
            "file_conversations.json",
            "canvases.json"
        }

        # First, count total JSON files to process
        total_files = 0
        json_files = []
        for subdir, dirs, files in os.walk(self.data_dir):
            subdir_path = Path(subdir)
            for fname in files:
                if fname.endswith('.json') and fname not in skip_files:
                    file_path = subdir_path / fname
                    try:
                        channel_path = subdir_path.relative_to(self.data_dir)
                        if len(channel_path.parts) >= 1:
                            json_files.append((subdir_path, fname))
                            total_files += 1
                    except ValueError:
                        continue

        # Initialize channels
        for subdir, dirs, files in os.walk(self.data_dir):
            for channel in dirs:
                self.messages[channel] = {}

        # Process JSON files with accurate progress bar
        with tqdm(total=total_files, desc="Processing Slack messages") as pbar:
            for subdir_path, fname in json_files:
                try:
                    # Get channel name from parent directory
                    channel_path = subdir_path.relative_to(self.data_dir)
                    cname = channel_path.parts[0]
                    date = fname.split(".")[-2]
                    
                    # Initialize channel dict if not exists
                    if cname not in self.messages:
                        self.messages[cname] = {}
                    
                    # Load messages
                    file_path = subdir_path / fname
                    with open(file_path, "r", encoding='utf-8') as f:
                        try:
                            messages = json.load(f)
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON in file: {file_path}")
                            pbar.update(1)
                            continue
                            
                    # Add metadata to messages
                    for m in messages:
                        m["date"] = date
                        m["channel"] = cname
                        
                    self.messages[cname][date] = messages
                    
                except Exception as e:
                    print(f"Error processing file {fname}: {str(e)}")
                finally:
                    pbar.update(1)

    def is_human_generated(self, message: Message) -> bool:
        # Check if the message is not a system message or bot message
        if 'subtype' in message:
            return False
        if 'bot_id' in message:
            return False
        return True

    def is_valid_message(self, message: Message) -> bool:
        # Check if the message has text content
        if 'text' not in message or not message['text'].strip():
            return False
        return True

    def get_messages(self, channel=None) -> List[Message]:
        m = []
        if channel is None:
            for c in self.messages:
                for d in self.messages[c]:
                    m += [msg for msg in self.messages[c][d] if self.is_human_generated(msg) and self.is_valid_message(msg)]
        else:
            if channel in self.messages:
                for d in self.messages[channel]:
                    m += [msg for msg in self.messages[channel][d] if self.is_human_generated(msg) and self.is_valid_message(msg)]
        return m

    def get_filtered_messages(self, filter_func: Callable[[Message], bool], channel=None) -> List[Message]:
        msgs = self.get_messages(channel)
        return [m for m in msgs if filter_func(m)]

    def get_messages_by_date(self, y: int, m: int, d: int) -> List[Message]:
        month = str(m).zfill(2)
        day = str(d).zfill(2)
        date = f"{y}-{month}-{day}"
        return self.get_filtered_messages(lambda m: m['date'] == date)

    def get_messages_by_datestring(self, date: str) -> List[Message]:
        return self.get_filtered_messages(lambda m: m['date'] == date)

    def get_messages_by_datestring_prefix(self, date: str) -> List[Message]:
        return self.get_filtered_messages(lambda m: m['date'].startswith(date))

    def get_messages_by_user(self, user: str) -> List[Message]:
        return self.get_filtered_messages(lambda m: 'user' in m and m['user'] == user)

    def get_channel_names(self) -> List[str]:
        return [c["name"] for c in self.channels]
