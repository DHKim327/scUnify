import numpy as np
import torch
from torch.utils.data import Dataset

from ...utils import load_yaml


class UCEDataset(Dataset):
    def __init__(self, adata, config):
        self.args = config
        
        # Seed for reproducibility
        seed = config.inference.get("seed", 42)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        _model_param = load_yaml(config._architecture_dir)[config.inference["nlayers"]]
        self.args.pad_length = _model_param["pad_length"]
        self.args.sample_size = _model_param["sample_size"]
        self.args.pad_length = _model_param["pad_length"]
        self.args.cls_token_idx = _model_param["cls_token_idx"]
        self.args.CHROM_TOKEN_OFFSET = _model_param["CHROM_TOKEN_OFFSET"]
        self.args.chrom_token_right_idx = _model_param["chrom_token_right_idx"]
        self.args.pad_token_idx = _model_param["pad_token_idx"]

        self.name = config.task_name
        self.adata, pe_row_idxs, chroms, starts = preproc(adata, config)
        self.X = self.adata.X
        self.num_cells, self.num_genes = self.X.shape
        if not isinstance(pe_row_idxs, torch.Tensor):
            pe_row_idxs = torch.as_tensor(pe_row_idxs, dtype=torch.long)

        self.dataset_to_protein_embeddings = {self.name: pe_row_idxs.long()}  # gene → token idx
        self.dataset_to_chroms = {self.name: np.asarray(chroms)}  # gene chrom codes
        self.dataset_to_starts = {self.name: np.asarray(starts)}  # gene start positions
        self.collator = Collator(self.args.pad_length)

    def __len__(self) -> int:
        return self.num_cells

    def __getitem__(self, idx: int):
        # 3) Read one row of counts -> weights (log1p) -> sequence sampling
        counts_row = _row_from_X(self.X, idx).astype(np.float32)
        counts = torch.from_numpy(counts_row).unsqueeze(0)  # (1, G)
        weights = torch.log1p(counts)
        s = torch.sum(weights)
        weights = weights / s if s.item() > 0 else torch.full_like(weights, 1.0 / weights.shape[-1])

        batch_sentences, mask, seq_len, cell_sentences = sample_cell_sentences(
            counts,
            weights,
            self.name,
            self.args,
            dataset_to_protein_embeddings=self.dataset_to_protein_embeddings,
            dataset_to_chroms=self.dataset_to_chroms,
            dataset_to_starts=self.dataset_to_starts,
        )
        # Maintain original return format + add cell_id
        return batch_sentences, mask, idx, seq_len, cell_sentences


class Collator:
    def __init__(self, pad_length):
        self.pad_length = pad_length

    def __call__(self, batch):
        batch_size = len(batch)
        batch_sentences = torch.zeros((batch_size, self.pad_length), dtype=torch.long)
        mask = torch.zeros((batch_size, self.pad_length), dtype=torch.float32)
        cell_sentences = torch.zeros((batch_size, self.pad_length), dtype=torch.long)

        idxs = torch.zeros(batch_size, dtype=torch.long)
        cell_ids = torch.zeros(batch_size, dtype=torch.long)

        i = 0
        max_len = 0
        for bs, msk, idx, seq_len, cs in batch:
            batch_sentences[i, :] = bs
            cell_sentences[i, :] = cs
            max_len = max(max_len, seq_len)
            mask[i, :] = msk
            idxs[i] = int(idx)
            i += 1

        return (
            batch_sentences[:, :max_len],
            mask[:, :max_len],
            idxs,
            cell_sentences[:, :max_len],
        )


def sample_cell_sentences(
    counts, batch_weights, dataset, args, dataset_to_protein_embeddings, dataset_to_chroms, dataset_to_starts, rng=None
):
    """
    Sample cell sentences for UCE model.
    
    Args:
        rng: numpy RandomState for reproducibility. If None, uses global np.random.
    """
    if rng is None:
        rng = np.random  # fallback to global state (controlled by worker_init_fn)
    
    dataset_idxs = dataset_to_protein_embeddings[dataset]  # get the dataset specific protein embedding idxs
    if isinstance(dataset_idxs, torch.Tensor):
        dataset_idxs = dataset_idxs.cpu().numpy()
    else:
        dataset_idxs = np.asarray(dataset_idxs)

    cell_sentences = torch.zeros((counts.shape[0], args.pad_length))  # init the cell representation as 0s
    mask = torch.zeros((counts.shape[0], args.pad_length))  # start of masking the whole sequence
    chroms = dataset_to_chroms[dataset]  # get the dataset specific chroms for each gene
    starts = dataset_to_starts[dataset]  # get the dataset specific genomic start locations for each gene

    longest_seq_len = 0  # we need to keep track of this so we can subset the batch at the end

    for c, cell in enumerate(counts):
        weights = batch_weights[c].numpy()
        weights = weights / sum(weights)  # RE NORM after mask

        # randomly choose the genes that will make up the sample, weighted by expression, with replacement
        choice_idx = rng.choice(np.arange(len(weights)), size=args.sample_size, p=weights, replace=True)
        choosen_chrom = chroms[choice_idx]  # get the sampled genes chromosomes
        # order the genes by chromosome
        chrom_sort = np.argsort(choosen_chrom)
        choice_idx = choice_idx[chrom_sort]

        # sort the genes by start
        new_chrom = chroms[choice_idx]
        choosen_starts = starts[choice_idx]

        ordered_choice_idx = np.full((args.pad_length), args.cls_token_idx, dtype=np.int64)  # start with cls
        # i= 0 first token is CLS
        i = 1  # continue on to the rest of the sequence with left bracket being assumed.
        # Shuffle the chroms now, there's no natural order to chromosomes
        uq_chroms = np.unique(new_chrom)
        rng.shuffle(uq_chroms)  # shuffle

        # This loop is actually just over one cell
        for chrom in uq_chroms:
            # Open Chrom token
            ordered_choice_idx[i] = (
                int(chrom) + args.CHROM_TOKEN_OFFSET
            )  # token of this chromosome # i = 1 next token is a chrom open
            i += 1
            # now sort the genes by start order within the chroms
            loc = np.where(new_chrom == chrom)[0]
            sort_by_start = np.argsort(choosen_starts[loc])  # start locations for this chromsome

            to_add = choice_idx[loc[sort_by_start]]
            ordered_choice_idx[i : (i + len(to_add))] = dataset_idxs[to_add]
            i += len(to_add)
            ordered_choice_idx[i] = args.chrom_token_right_idx  # add the chrom sep again
            i += 1  # add the closing token again

        longest_seq_len = max(longest_seq_len, i)
        remainder_len = args.pad_length - i

        cell_mask = torch.concat(
            (
                torch.ones(i),
                # pay attention to all of these tokens, ignore the rest!
                torch.zeros(remainder_len),
            )
        )

        mask[c, :] = cell_mask

        ordered_choice_idx[i:] = args.pad_token_idx  # the remainder of the sequence
        cell_sentences[c, :] = torch.from_numpy(ordered_choice_idx)

    cell_sentences_pe = cell_sentences.long()  # token indices

    return cell_sentences_pe, mask, longest_seq_len, cell_sentences


def _row_from_X(X, i: int) -> np.ndarray:
    if hasattr(X, "tocsr"):
        return X[i, :].toarray().ravel()
    arr = np.asarray(X)
    return arr[i].ravel()


############# For preprocessing

import pickle

import pandas as pd
import scanpy as sc


def preproc(adata, cfgs):
    # 1) Filter in same order as UCE (genes -> cells)
    if cfgs.preprocessing["filter"]:
        sc.pp.filter_genes(adata, min_cells=cfgs.preprocessing["filter_genes"])
        sc.pp.filter_cells(adata, min_genes=cfgs.preprocessing["filter_cells"])

    # 2) Subset embedding genes using UCE method (must use this function)
    adata = _preproc_raw_adata(adata, cfgs)
    # 3) HVG is not used (UCE default)
    #    => keep hv_genes=None

    # 4) Create indices (function below follows original UCE logic)

    species_to_pe = {
        specie: torch.load(cfgs.resources["protein_embeddings"][specie]) for specie in [cfgs.preprocessing["species"]]
    }
    species_to_pe = {species: {k.upper(): v for k, v in pe.items()} for species, pe in species_to_pe.items()}

    with open(cfgs.resources["offset_pkl_path"], "rb") as f:
        species_to_offsets = pickle.load(f)
    gene_to_chrom_pos = pd.read_csv(cfgs.resources["spec_chrom_csv_path"])
    gene_to_chrom_pos["spec_chrom"] = pd.Categorical(
        gene_to_chrom_pos["species"] + "_" + gene_to_chrom_pos["chromosome"]
    )

    spec_pe_genes = list(species_to_pe[cfgs.preprocessing["species"]].keys())  # Same as UCE
    offset = species_to_offsets[cfgs.preprocessing["species"]]

    pe_row_idxs, dataset_chroms, dataset_pos = adata_path_to_prot_chrom_starts(
        adata, cfgs.preprocessing["species"], spec_pe_genes, gene_to_chrom_pos, offset
    )

    return adata, pe_row_idxs, dataset_chroms, dataset_pos


def _preproc_raw_adata(adata, cfgs):
    species = cfgs.preprocessing["species"]
    species_to_gene_symbol_to_embedding = {
        specie: {
            gene_symbol.lower(): gene_embedding
            for gene_symbol, gene_embedding in torch.load(cfgs.resources["protein_embeddings"][specie]).items()
        }
        for specie in [species]
    }

    # Determine which genes to include based on gene expression and embedding availability
    genes_with_embeddings = set.intersection(
        *[set(gene_symbol_to_embedding) for gene_symbol_to_embedding in species_to_gene_symbol_to_embedding.values()]
    )
    genes_to_use = {gene for gene in adata.var_names if gene.lower() in genes_with_embeddings}
    # Subset data to only use genes with embeddings
    adata = adata[:, adata.var_names.isin(genes_to_use)]

    # Set up dictionary mapping species to gene embedding matrix (num_genes, embedding_dim)
    protein_embeddings = {
        specie: torch.stack(
            [species_to_gene_symbol_to_embedding[specie][gene_symbol.lower()] for gene_symbol in adata.var_names]
        )
        for specie in [species]
    }
    if cfgs.preprocessing["hv_genes"] is not None:
        sc.pp.highly_variable_genes(
            adata, flavor="seurat_v3", n_top_genes=cfgs.preprocessing["hv_genes"]
        )  # Expects Count Data

        hv_index = adata.var["highly_variable"]
        adata = adata[:, hv_index]  # Subset to hv genes only

    return adata


def adata_path_to_prot_chrom_starts(adata, dataset_species, spec_pe_genes, gene_to_chrom_pos, offset):
    """
    Given a :path: to an h5ad,
    """
    pe_row_idxs = torch.tensor([spec_pe_genes.index(k.upper()) + offset for k in adata.var_names]).long()

    spec_chrom = gene_to_chrom_pos[gene_to_chrom_pos["species"] == dataset_species].set_index("gene_symbol")

    gene_chrom = spec_chrom.loc[[k.upper() for k in adata.var_names]]

    dataset_chroms = gene_chrom["spec_chrom"].cat.codes  # now this is correctely indexed by species and chromosome
    dataset_pos = gene_chrom["start"].values
    return pe_row_idxs, dataset_chroms, dataset_pos
