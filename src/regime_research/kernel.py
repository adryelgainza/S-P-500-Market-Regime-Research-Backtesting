"""Fresh-kernel execution restricted to the runner's exact saved cell sequence."""

import hashlib
import json
import os
from pathlib import Path

from ipykernel.ipkernel import IPythonKernel
from ipykernel.kernelapp import IPKernelApp

from .experiments import canonical_bytes, content_hash, utc_now, ZERO_HASH


class AuditedKernel(IPythonKernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._plan = json.loads(Path(os.environ["REGIME_CELL_PLAN"]).read_text(encoding="utf-8"))
        self._audit_path = Path(os.environ["REGIME_KERNEL_AUDIT"])
        self._position = 0
        self._previous = ZERO_HASH
        self._active_code = None
        self._root_calls = 0
        self._violation = False
        self.shell.events.register("pre_run_cell", self._before_cell)

    def _record(self, kind, **values):
        event = {"Type": kind, "Recorded_UTC": utc_now(), "Previous_SHA256": self._previous, **values}
        event["Event_SHA256"] = content_hash(event)
        self._previous = event["Event_SHA256"]
        with self._audit_path.open("ab") as stream:
            stream.write(canonical_bytes(event)+b"\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _before_cell(self, info):
        self._root_calls += 1
        if self._active_code is None or info.raw_cell != self._active_code or self._root_calls != 1:
            self._violation = True
            self._record("unexpected_nested_execution", Source_SHA256=hashlib.sha256(info.raw_cell.encode()).hexdigest())

    async def do_execute(self, code, silent, store_history=True, user_expressions=None, allow_stdin=False, *, cell_meta=None, cell_id=None):
        if (self._position >= len(self._plan) or code != self._plan[self._position]["Source"]
                or silent or not store_history or user_expressions):
            self._violation = True
            self._record("unexpected_execute_request", Source_SHA256=hashlib.sha256(code.encode()).hexdigest())
            return {"status": "error", "ename": "ExecutionReceiptError", "evalue": "Execution differs from the saved plan",
                    "traceback": [], "execution_count": self.execution_count}
        planned = self._plan[self._position]
        self._active_code, self._root_calls = code, 0
        self._record("cell_started", Index=self._position, Cell_ID=planned["Cell_ID"], Source_SHA256=planned["Source_SHA256"])
        try:
            result = await super().do_execute(code, silent, store_history=store_history, user_expressions=user_expressions,
                allow_stdin=allow_stdin, cell_meta=cell_meta, cell_id=cell_id)
        finally:
            self._active_code = None
        status = "cell_completed" if result["status"] == "ok" and not self._violation else "cell_failed"
        self._record(status, Index=self._position, Cell_ID=planned["Cell_ID"], Source_SHA256=planned["Source_SHA256"],
                     Execution_Count=result["execution_count"])
        self._position += 1
        if self._violation:
            result.update(status="error", ename="ExecutionReceiptError", evalue="Unexpected nested execution", traceback=[])
        return result


if __name__ == "__main__":
    IPKernelApp.launch_instance(kernel_class=AuditedKernel)
