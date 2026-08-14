from pathlib import Path


def test_only_g1_teacher_ids_are_registered_in_source():
    source = (
        Path(__file__).resolve().parents[2]
        / "source/unitracker_lab/unitracker_lab/tasks/manager_based/unitracker_teacher/__init__.py"
    ).read_text(encoding="utf-8")
    assert 'id="Unitracker_Teacher-v0"' in source
    assert 'id="Unitracker_Teacher-Play-v0"' in source
    task_dir = (
        Path(__file__).resolve().parents[2]
        / "source/unitracker_lab/unitracker_lab/tasks/manager_based/unitracker_teacher"
    )
    assert {path.name for path in task_dir.iterdir() if path.is_dir()} <= {"agents", "mdp", "__pycache__"}
