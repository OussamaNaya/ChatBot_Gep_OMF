using System.Diagnostics;
using Microsoft.AspNetCore.Mvc;
using QA_ASP.Models;
using System.Net.Http;
using System.Text;
using Newtonsoft.Json;



namespace QA_ASP.Controllers;

public class HomeController : Controller
{
    private readonly ILogger<HomeController> _logger;

    public HomeController(ILogger<HomeController> logger)
    {
        _logger = logger;
    }

    public IActionResult Index()
    {
        return View();
    }

    public IActionResult Privacy()
    {
        return View();
    }

    public async Task<string> ChatAsync(string message)
    {
        if (string.IsNullOrWhiteSpace(message))
        {
            return "Veuillez entrer un message !";
        }

        string apiUrl = "http://localhost:8000/query"; // Remplace par l’URL réelle de ton API

        var requestData = new
        {
            question = message,
            role = "Simple" // Ou tout autre rôle que tu veux passer
        };

        using (var client = new HttpClient())
        {
            var content = new StringContent(JsonConvert.SerializeObject(requestData), Encoding.UTF8, "application/json");

            var response = await client.PostAsync(apiUrl, content);

            if (response.IsSuccessStatusCode)
            {
                var jsonResponse = await response.Content.ReadAsStringAsync();
                dynamic result = JsonConvert.DeserializeObject(jsonResponse);
                return result.answer;
            }
            else
            {
                return "Erreur lors de la communication avec le serveur.";
            }
        }
    }


    [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
    public IActionResult Error()
    {
        return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
    }
}
