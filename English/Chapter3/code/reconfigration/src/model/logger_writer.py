import logging

class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, message):
        # 受け取ったメッセージをバッファに追加
        self._buffer += message
        # バッファ内に改行があるか確認し、あれば行ごとにログ出力する
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():  # 空行は無視する
                self.logger.log(self.level, line)

    def flush(self):
        if self._buffer.strip():
            self.logger.log(self.level, self._buffer.strip())
        self._buffer = ""
