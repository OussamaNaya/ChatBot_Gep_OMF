using Microsoft.AspNetCore.Mvc;
using System.Text.Json;
using System.Diagnostics;

namespace ChatBot_GPT_OMF.Controllers
{
    public class QAController : Controller
    {
        private readonly ILogger<QAController> _logger;

        public QAController(ILogger<QAController> logger)
        {
            _logger = logger;
        }

        // GET: /QA/
        public IActionResult Index()
        {
            return View();
        }

        
    }
}
