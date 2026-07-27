/-
  Nostr provenance binding for Seiso attestations.

  Models the digests-only attestation (kind 31250) that seals a
  `dataset_merkle_root`, and the end-to-end claim checked by
  `seiso provenance dataset-verify-proof` when a relay event is used:

      valid NIP-01 event signature
        ∧ membership path reaches the sealed root
        ∧ proof.run_id matches the attestation

  BIP-340 Schnorr soundness is assumed as an axiom (not re-proved here).
  Merkle path completeness is proved in `Merkle.lean`.
-/

import SeisoProvenance.Merkle

namespace SeisoProvenance.Nostr

open SeisoProvenance.Merkle

/-- Digest carrier for SHA-256 outputs (32-byte values in the implementation).
    We use `Nat` as an abstract identifier space; only extensional equalities
    from `nodeHash` / `leafHash` matter in the proofs. -/
abbrev Digest := Nat

/-- Opaque hash used for leaves, internal nodes, and NIP-01 event ids. -/
opaque sha256 (msg : List UInt8) : Digest

/-- Domain-separated node combiner matching `_node_hash`. -/
opaque nodeHash (left right : Digest) : Digest

/-- Domain-separated leaf hash matching `_leaf_hash`. -/
opaque leafHash (runId fingerprint : String) : Digest

/-- Seiso addressable provenance kind (NIP-01: 30000 ≤ n < 40000). -/
def provenanceKind : Nat := 31250

/-- Digests-only attestation payload (`seiso.provenance.attestation/v2`). -/
structure AttestationV2 where
  schema : String := "seiso.provenance.attestation/v2"
  pipeline : String
  runId : String
  datasetMerkleRoot : Digest
  datasetMerkleLeafCount : Nat
  datasetMerkleAlg : String := "seiso.dataset.merkle/v1"
  deriving Repr

/-- Wire membership proof (`seiso.dataset.membership_proof/v1`). -/
structure MembershipProof where
  schema : String := "seiso.dataset.membership_proof/v1"
  runId : String
  fingerprint : String
  root : Digest
  path : List (PathStep Digest)
  deriving Repr

/-- NIP-01 event carrying an attestation in `content`. -/
structure ProvenanceEvent where
  id : Digest
  pubkey : Digest
  createdAt : Nat
  kind : Nat
  dTag : String
  content : AttestationV2
  sig : Digest  -- 64-byte Schnorr sig represented abstractly
  deriving Repr

/-- Expected addressable `d` tag: `pipeline:run_id`. -/
def expectedDTag (att : AttestationV2) : String :=
  att.pipeline ++ ":" ++ att.runId

/-- BIP-340 verification oracle (assumed sound for formal binding). -/
opaque validSchnorr (pubkey msg sig : Digest) : Prop

/-- Abstract NIP-01 id: SHA-256 of the canonical serialization preimage. -/
opaque eventIdOf (pubkey : Digest) (createdAt kind : Nat)
    (dTag : String) (content : AttestationV2) : Digest

/-- Event well-formedness: kind, d-tag, id binding, and Schnorr signature. -/
def eventValid (e : ProvenanceEvent) : Prop :=
  e.kind = provenanceKind ∧
  e.dTag = expectedDTag e.content ∧
  e.id = eventIdOf e.pubkey e.createdAt e.kind e.dTag e.content ∧
  validSchnorr e.pubkey e.id e.sig

/-- Local Merkle check used by `dataset-verify-proof --local-only`. -/
def membershipPathValid (proof : MembershipProof) : Prop :=
  pathReaches nodeHash (leafHash proof.runId proof.fingerprint) proof.root proof.path

/-- Full Nostr-backed membership claim:
    signed attestation seals `root`, and the path reaches that root. -/
def nostrMembershipValid (e : ProvenanceEvent) (proof : MembershipProof) : Prop :=
  eventValid e ∧
  proof.root = e.content.datasetMerkleRoot ∧
  proof.runId = e.content.runId ∧
  membershipPathValid proof

/-- Honest corpus commitment: leaf digests under a run id. -/
def commitLeaves (runId : String) (fingerprints : List String) : List Digest :=
  fingerprints.map (leafHash runId)

theorem atx_map {α β : Type} [Inhabited α] [Inhabited β]
    (f : α → β) (xs : List α) (i : Nat) (hi : i < xs.length) :
    atx (xs.map f) i = f (atx xs i) := by
  induction xs generalizing i with
  | nil => cases hi
  | cons x t ih =>
    cases i with
    | zero => simp [atx, List.map]
    | succ j =>
      simp only [List.map, List.length_cons, atx] at *
      exact ih j (by omega)

/-- If a fingerprint is in the committed corpus and a valid event seals that
    corpus root, the membership proof opened for that fingerprint satisfies
    the Nostr-backed verifier. -/
theorem nostr_membership_complete
    (e : ProvenanceEvent)
    (runId : String)
    (fingerprints : List String)
    (idx : Nat)
    (hne : fingerprints ≠ [])
    (hidx : idx < fingerprints.length)
    (hleaves :
      e.content.datasetMerkleRoot = root nodeHash (commitLeaves runId fingerprints) ∧
      e.content.runId = runId ∧
      e.content.datasetMerkleLeafCount = fingerprints.length)
    (hevent : eventValid e) :
    nostrMembershipValid e
      { runId := runId
        fingerprint := atx fingerprints idx
        root := root nodeHash (commitLeaves runId fingerprints)
        path := openPath nodeHash (commitLeaves runId fingerprints) idx } := by
  refine ⟨hevent, ?_, ?_, ?_⟩
  · exact hleaves.1.symm
  · exact hleaves.2.1.symm
  ·
    let leaves := commitLeaves runId fingerprints
    have hneL : leaves ≠ [] := by
      change commitLeaves runId fingerprints ≠ []
      simp [commitLeaves]
      exact hne
    have hidxL : idx < leaves.length := by
      change idx < (commitLeaves runId fingerprints).length
      simp [commitLeaves]
      exact hidx
    have hfp :
        leafHash runId (atx fingerprints idx) =
          atx (commitLeaves runId fingerprints) idx := by
      simp [commitLeaves, atx_map (leafHash runId) fingerprints idx hidx]
    have hreach :=
      openPath_reaches_root (α := Digest) nodeHash
        (commitLeaves runId fingerprints) idx hneL hidxL
    simpa [membershipPathValid, hfp, leaves] using hreach

/-- Structural consequence of a successful Nostr-backed membership check:
    the event is valid and the path folds to the sealed root. -/
theorem nostr_membership_implies_path_to_sealed_root
    (e : ProvenanceEvent) (proof : MembershipProof)
    (h : nostrMembershipValid e proof) :
    eventValid e ∧
      pathReaches nodeHash (leafHash proof.runId proof.fingerprint)
        e.content.datasetMerkleRoot proof.path ∧
      proof.runId = e.content.runId := by
  rcases h with ⟨he, hroot, hrun, hpath⟩
  refine ⟨he, ?_, hrun⟩
  simpa [membershipPathValid, hroot] using hpath

end SeisoProvenance.Nostr
