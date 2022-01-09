using TorchSharp;
using static TorchSharp.torch;
using static System.Linq.Enumerable;


var x = tensor(new[] { 1, 2, 3 }, new[] { 3L, 1 });
var y = tensor(new[] { 2, 4, 6 }, new[] { 3L, 1 });

var W = zeros(1, requiresGrad: true);
var b = zeros(1, requiresGrad: true);

var optimizer = optim.SGD(new[] { W, b }, learningRate: 0.01);
var nb_epochs = 2000;
foreach (var epoch in Range(1, nb_epochs))
{
    var hypothesis = x * W + b;
    var cost = mean(pow(hypothesis - y, 2));
    optimizer.zero_grad();
    cost.backward();
    optimizer.step();

    if (epoch % 100 == 0)
        Console.WriteLine($"Epoch: {epoch}, W: {W.ToSingle()}, b: {b.ToSingle()}, Cost: {cost.ToSingle()}");
}
//Console.WriteLine(mytensor(new List<List<float>> { new List<float> { 1, 2 } }).ToString(true));
Console.WriteLine(new test().mytensor(new List<List<List<float>>> { new List<List<float>> { new List<float> { 1, 2 } } }).ToString(true));
//Console.WriteLine(typeof(List<List<float>>) is IList<List<float>>);

class test
{
    public Tensor mytensor<T>(IList<T> t)
    {
        if(!typeof(T).IsGenericType)
        {
            switch(t)
            {
                case IList<bool> t1:
                    return tensor(t1);
                case IList<float> t1:
                    return tensor(t1);
                case IList<int> t1:
                    return tensor(t1);
                case IList<byte> t1:
                    return tensor(t1);
                case IList<sbyte> t1:
                    return tensor(t1);
                case IList<short> t1:
                    return tensor(t1);
                case IList<long> t1:
                    return tensor(t1);
                case IList<double> t1:
                    return tensor(t1);
                default:
                    throw new ArrayTypeMismatchException($"Type of {typeof(T)} is not supported");
            }
        }
        else if (typeof(T).GetGenericTypeDefinition() == typeof(List<>))
        {
            var tensors = new List<Tensor>();
            foreach (var k in t)
            {
                tensors.Add(((Tensor)GetType().GetMethod(nameof(mytensor)).MakeGenericMethod(typeof(T).GetGenericArguments()[0]).Invoke(this, new[] { (object)k })).unsqueeze(0));
            }
            return cat(tensors, 0);
        }
        else throw new ArrayTypeMismatchException($"Type of {typeof(T)} is not supported");
    }
}


//Console.WriteLine("Generic: " + GetType().GetMethod(nameof(mytensor)).MakeGenericMethod(typeof(T).GetGenericArguments()[0]));
                