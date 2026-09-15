"""Offline performance models built through Balsa; no inference policy in Balsa."""
from std.sys import argv, size_of
from balsa import ModelBuilder, TreeBuilder, TaskType, save
from numojo import NDArray
from numojo.core.type_aliases import Shape
from gpu_samples import fill_samples, mix


def generate[dtype: DType](args: List[String]) raises:
    var trees = Int(args[3])
    var depth = Int(args[4])
    var features = Int(args[5])
    var seed = Int(args[6])
    var uneven = args[7] == "uneven"
    var binary = args[8] == "binary"
    if (
        trees < 1
        or trees > 1024
        or depth < 1
        or depth > 10
        or features < 1
        or features > 512
    ):
        raise Error("Corpus dimensions exceed generation limits")
    if trees * ((1 << (depth + 1)) - 1) > 262144:
        raise Error("Corpus exceeds total node cap")
    var builder = ModelBuilder[dtype](
        Int32(features),
        task_type=TaskType.BINARY_CLF if binary else TaskType.REGRESSOR,
        postprocessor="sigmoid" if binary else "identity",
        base_scores=[0.125],
    )
    for t in range(trees):
        # Uneven trees retain the requested maximum depth but vary by tree;
        # within a tree, half the terminal subtrees stop one level earlier.
        var d = depth - (t % depth) if uneven else depth
        var nodes = (1 << (d + 1)) - 1
        if uneven and d > 1:
            nodes -= 1 << (d - 1)
        var tree = TreeBuilder[dtype](nodes)
        var internal = (nodes - 1) // 2
        for n in range(nodes):
            var h = mix(UInt32(seed + t * 65537 + n))
            if n < internal:
                tree.numerical_split(
                    n,
                    feature=Int32(h % UInt32(features)),
                    threshold=Scalar[dtype](
                        Float64(Int(mix(h) % 2048) - 1024) / 1024
                    ),
                    left=Int32(2 * n + 1),
                    right=Int32(2 * n + 2),
                    default_left=Bool(h & 1),
                )
            else:
                tree.leaf(n, Scalar[dtype](Float64(Int(h % 33) - 16) / 256))
        builder.add_tree(tree^.build())
    var model = builder^.build()
    save(model, args[9])


def samples[dtype: DType](args: List[String]) raises:
    var rows = Int(args[3])
    var features = Int(args[4])
    if rows < 0 or rows > 65536 or features < 1 or features > 512:
        raise Error("Sample dimensions exceed corpus cap")
    var input = NDArray[dtype](Shape(rows, features))
    fill_samples(input, Int(args[5]), Int(args[6]), args[7])
    with open(args[8], "w") as file:
        file.write_all(
            Span(
                unsafe_ptr=input.unsafe_ptr().unsafe_bitcast[UInt8](),
                length=input.size * size_of[Scalar[dtype]](),
            )
        )


def main() raises:
    var args = List[String]()
    for arg in argv():
        args.append(String(arg))
    if len(args) < 3:
        raise Error("Usage: generator model|samples DTYPE ...")
    if args[1] == "model" and len(args) == 10:
        if args[2] == "float32":
            generate[DType.float32](args)
        elif args[2] == "float64":
            generate[DType.float64](args)
        else:
            raise Error("Unsupported dtype")
    elif args[1] == "samples" and len(args) == 9:
        if args[2] == "float32":
            samples[DType.float32](args)
        elif args[2] == "float64":
            samples[DType.float64](args)
        else:
            raise Error("Unsupported dtype")
    else:
        raise Error("Invalid generator arguments")
