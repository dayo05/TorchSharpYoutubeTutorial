using System.Net;
using System.Net.Sockets;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;
using static System.Linq.Enumerable;

var socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
socket.Bind(new IPEndPoint(IPAddress.Loopback, 8080));
socket.Listen(10);

var client = socket.Accept();

var xList = new[] { 1, 2, 3 };
var yList = new[] { 3, 5, 7 };

foreach (var d in Range(0, xList.Length).Select(x => (xList[x], yList[x])))
{
    SendMessage($"P{d.Item1},{d.Item2}"); 
    Console.WriteLine($"P{d.Item1},{d.Item2}");
}

var x = tensor(xList, new[] { 3L, 1 });
Console.WriteLine(x.ToString(true));
var y = tensor(yList, new[] { 3L, 1 });
Console.WriteLine(y.ToString(true));

var W = zeros(1, requiresGrad: true);
Console.WriteLine(W.ToString(true));
var b = zeros(1, requiresGrad: true);
Console.WriteLine(b.ToString(true));

var optimizer = optim.SGD(new[] { W, b }, learningRate: 0.01);
var nb_epochs = 200;
foreach (var epoch in Range(1, nb_epochs))
{
    var hypothesis = x * W + b;
    var cost = mean(pow(hypothesis - y, 2));
    optimizer.zero_grad();
    cost.backward();
    optimizer.step();

    if (epoch % 1 == 0)
    {
        SendMessage($"{W.ToSingle()},{b.ToSingle()}");
        Console.WriteLine($"Epoch: {epoch}, W: {W.ToSingle()}, b: {b.ToSingle()}, Cost: {cost.ToSingle()}");
    }
}

SendMessage("end\n");
client.Close();


void SendMessage(string data)
    => client.Send(Encoding.UTF8.GetBytes(data + "\n"));
