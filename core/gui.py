# core/gui.py
import sys
from pathlib import Path

try:
    from qtpy.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox, QCheckBox, QSpinBox
    )
    from qtpy.QtCore import Qt
    QT_AVAILABLE = True
except Exception:  # pragma: no cover
    QT_AVAILABLE = False

def run_gui(comfy_path: Path | None = None,
            workflow_path: Path | None = None,
            extract_minimal: bool = False,
            port: int = 8000,
            generations: int = 10,
            num_instances: int = 1,
            run_default: bool = False) -> dict:
    """
    Show a Qt dialog to pick options.

    Returns
    -------
    dict with keys: 'comfy_path', 'workflow_path', 'extract_minimal', 'port', 'generations', 'num_instances', 'run_default'
    """
    if not QT_AVAILABLE:  # pragma: no cover
        print("Qt bindings not found – falling back to CLI mode.")
        sys.exit(0)

    app = QApplication(sys.argv)
    win = QWidget()
    win.setWindowTitle("ComfyUI Benchmark – Select Options")
    win.setFixedSize(620, 300)

    layout = QVBoxLayout()
    win.setLayout(layout)

    # ------------------------------------------------------------------
    # ComfyUI folder
    # ------------------------------------------------------------------
    comfy_row = QHBoxLayout()
    comfy_row.addWidget(QLabel("ComfyUI folder (-c):"))
    comfy_edit = QLineEdit(str(comfy_path) if comfy_path else "")
    comfy_row.addWidget(comfy_edit)

    def browse_comfy():
        folder = QFileDialog.getExistingDirectory(
            win,
            "Select ComfyUI folder",
            comfy_edit.text() or str(Path.home()))
        if folder:
            comfy_edit.setText(folder)

    btn = QPushButton("Browse…")
    btn.clicked.connect(browse_comfy)
    comfy_row.addWidget(btn)
    layout.addLayout(comfy_row)

    # ------------------------------------------------------------------
    # Workflow (ZIP or JSON)
    # ------------------------------------------------------------------
    workflow_row = QHBoxLayout()
    workflow_row.addWidget(QLabel("Workflow (-w) – ZIP or .json file:"))
    workflow_edit = QLineEdit(str(workflow_path) if workflow_path else "")
    workflow_row.addWidget(workflow_edit)

    use_folder_check = QCheckBox("Use Folder (for .json)")
    use_folder_check.setChecked(False)
    workflow_row.addWidget(use_folder_check)

    def browse_workflow():
        file, _ = QFileDialog.getOpenFileName(
            win,
            "Select workflow ZIP or .json",
            workflow_edit.text() or str(Path.home()),
            "Workflow files (*.zip *.json);;All files (*)")
        if file:
            workflow_edit.setText(file)

    btn = QPushButton("Browse…")
    btn.clicked.connect(browse_workflow)
    workflow_row.addWidget(btn)
    layout.addLayout(workflow_row)

    # ------------------------------------------------------------------
    # Minimal Extraction checkbox
    # ------------------------------------------------------------------
    minimal_check = QCheckBox("Minimal Extraction (Models/Nodes must be previously installed) (-e)")
    minimal_check.setChecked(extract_minimal)
    layout.addWidget(minimal_check)

    # ------------------------------------------------------------------
    # Port spinbox
    # ------------------------------------------------------------------
    port_row = QHBoxLayout()
    port_row.addWidget(QLabel("Port (-p):"))
    port_spin = QSpinBox()
    port_spin.setRange(1024, 65535)
    port_spin.setValue(port)
    port_row.addWidget(port_spin)
    port_row.addStretch()
    layout.addLayout(port_row)

    # ------------------------------------------------------------------
    # Generations and Instances spinboxes
    # ------------------------------------------------------------------
    spins_row = QHBoxLayout()
    spins_row.addWidget(QLabel("Number of Image Generations (-g):"))
    generations_spin = QSpinBox()
    generations_spin.setRange(1, 1000)
    generations_spin.setValue(generations)
    spins_row.addWidget(generations_spin)

    spins_row.addWidget(QLabel("Number of concurrent sessions (-n):"))
    num_instances_spin = QSpinBox()
    num_instances_spin.setRange(1, 100)
    num_instances_spin.setValue(num_instances)
    spins_row.addWidget(num_instances_spin)
    layout.addLayout(spins_row)

    # ------------------------------------------------------------------
    # Use Package Defaults checkbox
    # ------------------------------------------------------------------
    default_check = QCheckBox("Use Package Defaults (-r)")
    default_check.setChecked(run_default)

    def toggle_spins(checked: bool):
        generations_spin.setEnabled(not checked)
        num_instances_spin.setEnabled(not checked)

    default_check.toggled.connect(toggle_spins)
    toggle_spins(run_default)  # Initial state
    layout.addWidget(default_check)

    # ------------------------------------------------------------------
    # OK / Cancel
    # ------------------------------------------------------------------
    btn_row = QHBoxLayout()
    btn_row.addStretch()

    ok_btn = QPushButton("OK")
    ok_btn.setDefault(True)

    def on_ok():
        c = Path(comfy_edit.text().strip())
        w = Path(workflow_edit.text().strip())
        use_folder = use_folder_check.isChecked()

        # ---- basic existence ----
        if not c.exists() or not c.is_dir():
            QMessageBox.critical(win, "Error", "ComfyUI folder does not exist or is not a directory.")
            return
        if not w.exists():
            QMessageBox.critical(win, "Error", "Workflow path does not exist.")
            return

        # ---- workflow-specific validation ----
        if w.suffix.lower() == ".json" and use_folder:
            w = w.parent  # Use folder
        if w.suffix.lower() != ".zip" and not (w / "workflow.json").exists():
            QMessageBox.critical(
                win, "Error",
                "Workflow folder must contain a file named 'workflow.json'.")
            return

        # ---- Save results ----
        win.result = {
            'comfy_path': c,
            'workflow_path': w,
            'extract_minimal': minimal_check.isChecked(),
            'port': port_spin.value(),
            'generations': generations_spin.value(),
            'num_instances': num_instances_spin.value(),
            'run_default': default_check.isChecked()
        }
        win.close()

    ok_btn.clicked.connect(on_ok)
    btn_row.addWidget(ok_btn)

    cancel_btn = QPushButton("Cancel")
    cancel_btn.clicked.connect(win.close)
    btn_row.addWidget(cancel_btn)

    layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Run modal dialog
    # ------------------------------------------------------------------
    win.result = None
    win.show()
    app.exec_()

    if win.result is None:
        print("GUI cancelled – exiting.")
        sys.exit(0)

    return win.result