from qbr.adapters.persistence.sqlalchemy_models import Base

def test_expected_tables_and_foreign_keys():
    assert set(Base.metadata.tables) == {'qbr_documents', 'qbr_fragment_outcomes', 'qbr_evidence'}
    fragment = Base.metadata.tables['qbr_fragment_outcomes']
    assert any(fk.target_fullname == 'qbr_documents.id' for fk in fragment.foreign_keys)
    assert 'source_id' in Base.metadata.tables['qbr_evidence'].c
