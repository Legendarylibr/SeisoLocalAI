/-
  Seiso dataset Merkle membership (`seiso/research/dataset_merkle.py`).

  List algorithm with odd-length duplication, path opening, and path folding
  matching `open_membership_path` / `verify_membership_path`.

  Completeness: an honestly opened path always folds back to the Merkle root,
  for an arbitrary binary combiner (Python: SHA-256 + domain tag).
-/

namespace SeisoProvenance.Merkle

/-- Strong induction on list length (Lean 4.16 stdlib has no `List.strongInductionOn`). -/
theorem list_strong_induction {α : Type} {P : List α → Prop}
    (H : ∀ xs, (∀ ys, ys.length < xs.length → P ys) → P xs) :
    ∀ xs, P xs := by
  intro xs
  have : ∀ n : Nat, ∀ ys : List α, ys.length = n → P ys := by
    intro n
    induction n using Nat.strongRecOn with
    | ind n ihn =>
      intro ys hlen
      apply H
      intro zs hlt
      have hz : zs.length < n := by omega
      exact ihn zs.length hz zs rfl
  exact this xs.length xs rfl

/-- Sibling orientation on the wire (`"L"` / `"R"` in proof JSON). -/
inductive Side where
  | L
  | R
  deriving DecidableEq, Repr, Inhabited

/-- One step of a membership path. -/
structure PathStep (α : Type) where
  side : Side
  sibling : α
  deriving Repr

/-- Total index into a list (defaults when out of range). -/
def atx {α : Type} [Inhabited α] : List α → Nat → α
  | [], _ => default
  | x :: _, 0 => x
  | _ :: xs, n + 1 => atx xs n

theorem atx_cons_cons {α} [Inhabited α] (a b : α) (u : List α) (j : Nat) :
    atx (a :: b :: u) (j + 2) = atx u j :=
  rfl

/-- Pair adjacent digests; duplicate the final node when the level length is odd. -/
def nextLevel {α : Type} (nodeHash : α → α → α) : List α → List α
  | [] => []
  | [x] => [nodeHash x x]
  | x :: y :: rest => nodeHash x y :: nextLevel nodeHash rest

theorem nextLevel_length {α} (h : α → α → α) :
    ∀ xs : List α, (nextLevel h xs).length = (xs.length + 1) / 2 := by
  apply list_strong_induction
  intro xs ih
  match xs with
  | [] => simp [nextLevel]
  | [x] => simp [nextLevel]
  | x :: y :: rest =>
    have hlt : rest.length < (x :: y :: rest).length := by
      simp only [List.length_cons]; omega
    simp only [nextLevel, List.length_cons, ih rest hlt]
    omega

theorem nextLevel_length_lt {α} (h : α → α → α) (xs : List α)
    (hlen : 2 ≤ xs.length) :
    (nextLevel h xs).length < xs.length := by
  have := nextLevel_length h xs
  omega

theorem nextLevel_ne_nil {α} (h : α → α → α) (xs : List α) (hne : xs ≠ []) :
    nextLevel h xs ≠ [] := by
  have hlen := nextLevel_length h xs
  have hpos : 0 < xs.length := by
    cases xs with
    | nil => exact (hne rfl).elim
    | cons _ _ => simp
  have : 0 < (nextLevel h xs).length := by omega
  intro empt
  simp [empt] at this

private theorem length_cons_cons_ge_two {α} (x y : α) (rest : List α) :
    2 ≤ (x :: y :: rest).length := by
  simp only [List.length_cons]
  exact Nat.le_add_left 2 rest.length

/-- Merkle root over a leaf list. -/
def root {α : Type} [Inhabited α] (nodeHash : α → α → α) : List α → α
  | [] => default
  | [x] => x
  | x :: y :: rest =>
      have : (nextLevel nodeHash (x :: y :: rest)).length <
          (x :: y :: rest).length :=
        nextLevel_length_lt nodeHash (x :: y :: rest)
          (length_cons_cons_ge_two x y rest)
      root nodeHash (nextLevel nodeHash (x :: y :: rest))
termination_by xs => xs.length

theorem root_of_cons_cons {α} [Inhabited α] (h : α → α → α)
    (x y : α) (rest : List α) :
    root h (x :: y :: rest) = root h (nextLevel h (x :: y :: rest)) := by
  -- The `have` proof term in `root` blocks `rfl`; unfold once.
  simp only [root]

/-- Sibling index and side for node index `idx` in a level of length `n`. -/
def siblingOf (n idx : Nat) : Side × Nat :=
  if idx % 2 = 0 then
    (Side.R, if idx + 1 < n then idx + 1 else idx)
  else
    (Side.L, idx - 1)

theorem siblingOf_lt (n idx : Nat) (hidx : idx < n) :
    (siblingOf n idx).2 < n := by
  unfold siblingOf
  by_cases heven : idx % 2 = 0
  · simp [heven]; split <;> omega
  · simp [heven]; omega

/-- Open a membership path for `leaves` at `index`. -/
def openPath {α : Type} [Inhabited α] (nodeHash : α → α → α)
    (leaves : List α) (index : Nat) : List (PathStep α) :=
  if hlen : leaves.length ≤ 1 then
    []
  else
    let pair := siblingOf leaves.length index
    let step : PathStep α := { side := pair.1, sibling := atx leaves pair.2 }
    step :: openPath nodeHash (nextLevel nodeHash leaves) (index / 2)
termination_by leaves.length
decreasing_by
  simp_wf
  exact nextLevel_length_lt nodeHash leaves (by omega)

/-- Fold a path upward from a leaf digest. -/
def foldPath {α : Type} (nodeHash : α → α → α) (current : α) :
    List (PathStep α) → α
  | [] => current
  | ⟨Side.L, sib⟩ :: rest => foldPath nodeHash (nodeHash sib current) rest
  | ⟨Side.R, sib⟩ :: rest => foldPath nodeHash (nodeHash current sib) rest

/-- Path verification predicate. -/
def pathReaches {α : Type} (nodeHash : α → α → α)
    (leaf expectedRoot : α) (path : List (PathStep α)) : Prop :=
  foldPath nodeHash leaf path = expectedRoot

/-- Parent covering `idx` via even/odd pairing. -/
def pairParent {α : Type} [Inhabited α] (nodeHash : α → α → α)
    (level : List α) (idx : Nat) : α :=
  let leftIdx := idx / 2 * 2
  let rightIdx := if leftIdx + 1 < level.length then leftIdx + 1 else leftIdx
  nodeHash (atx level leftIdx) (atx level rightIdx)

/-- Combined parent using `siblingOf` (matches one verification step). -/
def combine {α : Type} [Inhabited α] (nodeHash : α → α → α)
    (level : List α) (idx : Nat) : α :=
  let pair := siblingOf level.length idx
  match pair.1 with
  | Side.L => nodeHash (atx level pair.2) (atx level idx)
  | Side.R => nodeHash (atx level idx) (atx level pair.2)

/-- `nextLevel` at `idx/2` is the hash of the pair covering `idx`. -/
theorem nextLevel_atx_pair {α : Type} [Inhabited α] (nodeHash : α → α → α)
    (level : List α) (idx : Nat) (hidx : idx < level.length) :
    idx / 2 < (nextLevel nodeHash level).length ∧
      atx (nextLevel nodeHash level) (idx / 2) = pairParent nodeHash level idx := by
  have general :
      ∀ level : List α, ∀ idx : Nat, idx < level.length →
        idx / 2 < (nextLevel nodeHash level).length ∧
          atx (nextLevel nodeHash level) (idx / 2) =
            pairParent nodeHash level idx := by
    apply list_strong_induction
    intro level ih idx hidx
    match level with
    | [] => cases hidx
    | [x] =>
      have : idx = 0 := by simp at hidx; omega
      subst this
      simp [nextLevel, pairParent, atx]
    | x :: y :: rest =>
      match idx with
      | 0 =>
        refine ⟨?_, ?_⟩
        · have := nextLevel_length nodeHash (x :: y :: rest)
          simp at this ⊢; omega
        · simp [nextLevel, pairParent, atx]
      | 1 =>
        refine ⟨?_, ?_⟩
        · have := nextLevel_length nodeHash (x :: y :: rest)
          simp at this ⊢; omega
        · simp [nextLevel, pairParent, atx]
      | k + 2 =>
        have hlt : rest.length < (x :: y :: rest).length := by
          simp only [List.length_cons]; omega
        have hk : k < rest.length := by simp at hidx; omega
        have ⟨hpos, hval⟩ := ih rest hlt k hk
        refine ⟨?_, ?_⟩
        · have := nextLevel_length nodeHash (x :: y :: rest)
          simp at this ⊢; omega
        ·
          have hdiv : (k + 2) / 2 = k / 2 + 1 := by omega
          have hget :
              atx (nextLevel nodeHash (x :: y :: rest)) ((k + 2) / 2) =
                atx (nextLevel nodeHash rest) (k / 2) := by
            simp only [nextLevel, hdiv]
            rfl
          rw [hget, hval]
          unfold pairParent
          have hleft : (k + 2) / 2 * 2 = k / 2 * 2 + 2 := by omega
          simp only [List.length_cons, hleft]
          have L : atx (x :: y :: rest) (k / 2 * 2 + 2) = atx rest (k / 2 * 2) :=
            atx_cons_cons x y rest (k / 2 * 2)
          have Rcond :
              (if k / 2 * 2 + 2 + 1 < rest.length + 2 then k / 2 * 2 + 2 + 1
                else k / 2 * 2 + 2) =
                (if k / 2 * 2 + 1 < rest.length then k / 2 * 2 + 1
                  else k / 2 * 2) + 2 := by
            by_cases h1 : k / 2 * 2 + 2 + 1 < rest.length + 2
            · by_cases h2 : k / 2 * 2 + 1 < rest.length
              · simp [h1, h2]
              · simp [h1, h2]; omega
            · by_cases h2 : k / 2 * 2 + 1 < rest.length
              · simp [h1, h2]; omega
              · simp [h1, h2]
          have R :
              atx (x :: y :: rest)
                  (if k / 2 * 2 + 2 + 1 < rest.length + 2 then k / 2 * 2 + 2 + 1
                    else k / 2 * 2 + 2) =
                atx rest
                  (if k / 2 * 2 + 1 < rest.length then k / 2 * 2 + 1
                    else k / 2 * 2) := by
            rw [Rcond, atx_cons_cons]
          rw [L, R]
  exact general level idx hidx

/-- Combining with `siblingOf` matches the pair parent. -/
theorem combine_eq_pairParent {α : Type} [Inhabited α] (nodeHash : α → α → α)
    (level : List α) (idx : Nat) (hidx : idx < level.length) :
    combine nodeHash level idx = pairParent nodeHash level idx := by
  unfold combine siblingOf pairParent
  by_cases heven : idx % 2 = 0
  · have hleft : idx / 2 * 2 = idx := by omega
    simp [heven, hleft]
  · have hleft : idx / 2 * 2 = idx - 1 := by omega
    have hright : idx - 1 + 1 < level.length := by omega
    have hidx' : idx - 1 + 1 = idx := by omega
    simp [heven, hleft, hright, hidx', hidx]

/-- One climb step equals `nextLevel` at `idx/2`. -/
theorem climb_one_level {α : Type} [Inhabited α] (nodeHash : α → α → α)
    (level : List α) (idx : Nat) (hidx : idx < level.length) :
    combine nodeHash level idx = atx (nextLevel nodeHash level) (idx / 2) ∧
      idx / 2 < (nextLevel nodeHash level).length := by
  have ⟨hpos, hval⟩ := nextLevel_atx_pair nodeHash level idx hidx
  have hc := combine_eq_pairParent nodeHash level idx hidx
  exact ⟨hc.trans hval.symm, hpos⟩

theorem foldPath_step_eq {α : Type} [Inhabited α] (nodeHash : α → α → α)
    (level : List α) (idx : Nat) (rest : List (PathStep α))
    (_hidx : idx < level.length) :
    foldPath nodeHash (atx level idx)
        ({ side := (siblingOf level.length idx).1,
           sibling := atx level (siblingOf level.length idx).2 } :: rest) =
      foldPath nodeHash (combine nodeHash level idx) rest := by
  unfold combine
  cases hside : (siblingOf level.length idx).1 with
  | L => simp [foldPath, hside]
  | R => simp [foldPath, hside]

/-- Completeness: an honestly opened path folds back to the Merkle root. -/
theorem openPath_reaches_root {α : Type} [Inhabited α] (nodeHash : α → α → α)
    (leaves : List α) (idx : Nat) (hne : leaves ≠ []) (hidx : idx < leaves.length) :
    pathReaches nodeHash (atx leaves idx) (root nodeHash leaves)
      (openPath nodeHash leaves idx) := by
  have general :
      ∀ leaves : List α, ∀ idx : Nat, leaves ≠ [] → idx < leaves.length →
        pathReaches nodeHash (atx leaves idx) (root nodeHash leaves)
          (openPath nodeHash leaves idx) := by
    apply list_strong_induction
    intro leaves ih idx hne hidx
    match leaves with
    | [] => exact (hne rfl).elim
    | [x] =>
      have : idx = 0 := by simp at hidx; omega
      subst this
      simp [pathReaches, openPath, foldPath, root, atx]
    | x :: y :: rest =>
      have hlen : ¬ (x :: y :: rest).length ≤ 1 := by
        have := length_cons_cons_ge_two x y rest
        omega
      rw [pathReaches, openPath, dif_neg hlen]
      have ⟨hparent, hpos⟩ := climb_one_level nodeHash (x :: y :: rest) idx hidx
      rw [foldPath_step_eq nodeHash (x :: y :: rest) idx _ hidx, hparent,
        root_of_cons_cons]
      have hne_next : nextLevel nodeHash (x :: y :: rest) ≠ [] :=
        nextLevel_ne_nil nodeHash (x :: y :: rest) (List.cons_ne_nil _ _)
      have hlt : (nextLevel nodeHash (x :: y :: rest)).length <
          (x :: y :: rest).length :=
        nextLevel_length_lt nodeHash (x :: y :: rest)
          (length_cons_cons_ge_two x y rest)
      exact ih (nextLevel nodeHash (x :: y :: rest)) hlt (idx / 2) hne_next hpos
  exact general leaves idx hne hidx

/-- Domain tags from `dataset_merkle.py` (documentation; hash is abstract). -/
def leafDomainTag : String := "seiso.dataset.leaf/v1"
def nodeDomainTag : String := "seiso.dataset.node/v1"

end SeisoProvenance.Merkle
