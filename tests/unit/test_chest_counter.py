from src.chest_counter import ChestCounter
from src.capture.map_keys import GlobalMapKeys


def test_counts_clamp_at_zero_and_undo_including_new_round():
    counter = ChestCounter()
    assert not counter.apply('chest_gold_subtract') and not counter.can_undo
    counter.apply('chest_gold_add')
    counter.apply('chest_purple_add')
    counter.apply('chest_purple_add')
    counter.apply('chest_purple_subtract')
    assert (counter.gold, counter.purple) == (1, 1)
    assert counter.undo() and (counter.gold, counter.purple) == (1, 2)
    assert counter.reset() and (counter.gold, counter.purple) == (0, 0)
    assert counter.undo() and (counter.gold, counter.purple) == (1, 2)
    assert (ChestCounter().gold, ChestCounter().purple) == (0, 0)


def test_counter_raw_input_records_short_taps_and_foreground_at_press(monkeypatch):
    monkeypatch.setattr(GlobalMapKeys, '_read_held', staticmethod(lambda _: set()))
    source = GlobalMapKeys('G', ['IdentityV'], lambda: None)
    source.configure('G', ['IdentityV'], action_keys={'chest_gold_add': 'F5'})
    for down, title in ((True, 'IdentityV'), (True, 'IdentityV'), (False, 'Other'),
                        (True, 'Other'), (False, 'Other')):
        source._record(0x74, down, (42, title, 0))
    events = list(source.events.queue)
    assert [(event.action, event.down, event.enabled) for event in events] == [
        ('chest_gold_add', True, True), ('chest_gold_add', False, False),
        ('chest_gold_add', True, False), ('chest_gold_add', False, False)]
    source.configure('G', ['IdentityV'], suspended=True)
    source._record(0x74, True, (42, 'IdentityV', 0))
    assert not source.events.get_nowait().enabled


def test_rebinding_drops_old_events_and_requires_release_of_held_key(monkeypatch):
    monkeypatch.setattr(GlobalMapKeys, '_read_held', staticmethod(lambda _: set()))
    source = GlobalMapKeys('G', ['IdentityV'], lambda: None)
    source.configure('G', ['IdentityV'], action_keys={'chest_gold_add': 'F5'})
    source._record(0x74, True, (42, 'IdentityV', 0))
    monkeypatch.setattr(GlobalMapKeys, '_read_held', staticmethod(lambda _: {0x75}))
    source.configure('G', ['IdentityV'], action_keys={'chest_gold_add': 'F6'})
    source._record(0x74, True, (42, 'IdentityV', 0))
    source._record(0x75, True, (42, 'IdentityV', 0))
    assert source.events.empty()
    source._record(0x75, False, (42, 'IdentityV', 0))
    source._record(0x75, True, (42, 'IdentityV', 0))
    assert [event.down for event in source.events.queue] == [False, True]
